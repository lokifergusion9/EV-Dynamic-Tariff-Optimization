"""Stage 1 - Data preprocessing & feature engineering.

Turns the raw ACN-Data export (`acndata_sessions.json.xlsx`) into:
  * a clean session-level table  (one row per charging session)
  * an hourly site-level demand panel (the modeling table for the agents)

and loads the UrbanEV / ST-EVCDP dataset into a comparable hourly demand profile
for cross-dataset validation.

Every cleaning decision is logged so the assumptions are transparent (OP'26 brief).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config as C

# Real per-session fields begin at column index 6 (`_id`); columns 0-5 are
# JSON-export artifacts (`_meta, end, min_kWh, site, start, _items`).
SESSION_COLS = [
    "_id", "clusterID", "connectionTime", "disconnectTime", "doneChargingTime",
    "kWhDelivered", "sessionID", "siteID", "spaceID", "stationID", "timezone",
    "userID", "userInputs", "WhPerMile", "kWhRequested", "milesRequested",
    "minutesAvailable", "modifiedAt", "paymentRequired", "requestedDeparture",
    "userID2",
]

# Friendly names for known ACN sites (this export is Caltech, siteID 2).
SITE_NAMES = {"2": "caltech", "1": "jpl", "19": "office001"}


def _log(msg: str) -> None:
    print(f"[preprocess] {msg}")


# --------------------------------------------------------------------------- #
# Load + clean ACN sessions
# --------------------------------------------------------------------------- #
def load_acn_sessions(path=None) -> pd.DataFrame:
    """Load and clean the ACN session export into a tidy session-level frame."""
    path = path or C.ACN_XLSX
    _log(f"loading {path}")
    raw = pd.read_excel(path, engine="openpyxl", header=0)
    raw = raw.iloc[:, 6:]                       # drop the 6 JSON-artifact columns
    raw.columns = SESSION_COLS
    n0 = len(raw)

    # 1,305 rows are pagination/meta boundaries with no connectionTime -> drop.
    df = raw[raw["connectionTime"].notna() & (raw["connectionTime"] != "None")].copy()
    _log(f"dropped {n0 - len(df)} JSON pagination/meta rows -> {len(df)} candidate sessions")

    # Parse timestamps (RFC-1123 GMT strings) -> UTC, then to site-local time.
    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    tz = df["timezone"].mode().iat[0] if df["timezone"].notna().any() else "America/Los_Angeles"
    for col in ["connectionTime", "disconnectTime", "doneChargingTime"]:
        df[col + "_local"] = df[col].dt.tz_convert(tz)

    # Numeric coercion
    for col in ["kWhDelivered", "kWhRequested", "milesRequested", "WhPerMile",
                "minutesAvailable"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing-value handling (documented):
    #   - userInputs is null for 100% of rows -> drop column.
    #   - doneChargingTime missing -> fall back to disconnectTime (assume the car
    #     finished when it unplugged). Logged.
    df = df.drop(columns=["userInputs"])
    miss_done = df["doneChargingTime"].isna() & df["disconnectTime"].notna()
    _log(f"doneChargingTime missing for {int(miss_done.sum())} sessions -> filled with disconnectTime")
    df.loc[miss_done, "doneChargingTime_local"] = df.loc[miss_done, "disconnectTime_local"]

    # Durations (hours)
    conn = df["connectionTime_local"]
    df["connection_duration_h"] = (df["disconnectTime_local"] - conn).dt.total_seconds() / 3600
    df["charging_duration_h"] = (df["doneChargingTime_local"] - conn).dt.total_seconds() / 3600
    df["idle_h"] = df["connection_duration_h"] - df["charging_duration_h"]

    # Drop physically impossible / unusable sessions, logged at each step.
    before = len(df)
    df = df[df["connection_duration_h"] > 0]
    df = df[df["kWhDelivered"].fillna(0) >= 0]
    # clip charging duration into [0, connection] (sensor noise can overshoot)
    df["charging_duration_h"] = df["charging_duration_h"].clip(lower=0)
    df["charging_duration_h"] = np.minimum(df["charging_duration_h"], df["connection_duration_h"])
    df["idle_h"] = (df["connection_duration_h"] - df["charging_duration_h"]).clip(lower=0)
    _log(f"dropped {before - len(df)} sessions with non-positive duration / negative energy")

    # Calendar features (site-local)
    site_key = pd.to_numeric(df["siteID"], errors="coerce").astype("Int64").astype(str)
    df["site"] = site_key.map(SITE_NAMES).fillna("site_" + site_key)
    df["hour"] = conn.dt.hour
    df["dow"] = conn.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = conn.dt.month
    df["date"] = conn.dt.date
    df["hour_bucket"] = conn.dt.floor("h")

    # --- Engineered economic features (OP'26 brief) ------------------------- #
    # 1) Charger Utilization Rate = charging time / connection (occupied) time.
    df["charger_utilization"] = (df["charging_duration_h"] /
                                 df["connection_duration_h"]).clip(0, 1)
    # 3) Energy Cost per kWh = time-of-use grid procurement cost proxy.
    df["energy_cost_per_kwh"] = df["hour"].map(_tou_cost)
    # 2) Revenue per Session @ baseline tariff (kWh x INR/kWh).
    df["revenue_baseline"] = df["kWhDelivered"].fillna(0) * C.BASELINE_TARIFF
    df["gross_margin_baseline"] = (C.BASELINE_TARIFF - df["energy_cost_per_kwh"]) * df["kWhDelivered"].fillna(0)

    df = df.reset_index(drop=True)
    _log(f"final clean session count = {len(df)} (sites: {df['site'].value_counts().to_dict()})")
    return df


def _tou_cost(hour: int) -> float:
    if hour in C.TOU_PEAK_HOURS:
        return C.ENERGY_COST_PEAK
    if hour in C.TOU_OFFPEAK_HOURS:
        return C.ENERGY_COST_OFFPEAK
    return C.ENERGY_COST_SHOULDER


# --------------------------------------------------------------------------- #
# Concurrency-based features (Queue Length Proxy, Occupancy Density)
# --------------------------------------------------------------------------- #
def add_concurrency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute concurrent-session counts per site/hour using interval overlap.

    Queue Length Proxy  = number of simultaneously-active sessions in the hour.
    Occupancy Density   = active sessions / distinct stations seen at the site.

    Implemented by exploding each session over the hour buckets it spans, then
    counting per (site, hour) -- O(sessions x mean-span) instead of O(sessions^2).
    """
    df = df.copy()
    site_capacity = df.groupby("site")["stationID"].nunique().clip(lower=C.MIN_STATIONS_PER_SITE)

    spans = []
    conn_floor = df["connectionTime_local"].dt.floor("h")
    disc_floor = df["disconnectTime_local"].dt.floor("h")
    for site, cstart, cend in zip(df["site"], conn_floor, disc_floor):
        if pd.isna(cstart):
            continue
        if pd.isna(cend) or cend < cstart:
            cend = cstart
        for hb in pd.date_range(cstart, cend, freq="h"):
            spans.append((site, hb))
    counts = (pd.DataFrame(spans, columns=["site", "hour_bucket"])
                .value_counts().rename("queue_proxy").reset_index())
    counts["occupancy_density"] = counts["queue_proxy"] / counts["site"].map(site_capacity)

    df = df.merge(counts, on=["site", "hour_bucket"], how="left")
    df["queue_proxy"] = df["queue_proxy"].fillna(1)
    df["occupancy_density"] = df["occupancy_density"].fillna(0)
    return df


# --------------------------------------------------------------------------- #
# Hourly site-level demand panel (modeling table)
# --------------------------------------------------------------------------- #
def build_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sessions into a continuous hourly site-level panel."""
    site_capacity = df.groupby("site")["stationID"].nunique().clip(lower=C.MIN_STATIONS_PER_SITE)

    agg = (df.groupby(["site", "hour_bucket"])
             .agg(sessions=("sessionID", "count"),
                  energy_kwh=("kWhDelivered", "sum"),
                  mean_utilization=("charger_utilization", "mean"),
                  mean_charge_h=("charging_duration_h", "mean"),
                  mean_energy_cost=("energy_cost_per_kwh", "mean"))
             .reset_index())

    # Build a dense hourly grid per site so gaps become explicit zeros.
    panels = []
    for site, g in agg.groupby("site"):
        full = pd.date_range(g["hour_bucket"].min(), g["hour_bucket"].max(), freq="h")
        gi = g.set_index("hour_bucket").reindex(full)
        gi["site"] = site
        gi["sessions"] = gi["sessions"].fillna(0)
        gi["energy_kwh"] = gi["energy_kwh"].fillna(0)
        gi["mean_utilization"] = gi["mean_utilization"].fillna(0)
        gi["mean_charge_h"] = gi["mean_charge_h"].fillna(0)
        gi["mean_energy_cost"] = gi["mean_energy_cost"].fillna(gi["mean_energy_cost"].median())
        gi.index.name = "hour_bucket"
        panels.append(gi.reset_index())
    panel = pd.concat(panels, ignore_index=True)

    panel["capacity"] = panel["site"].map(site_capacity)
    # station-utilization: occupied station-hours / available station-hours
    panel["station_utilization"] = (panel["sessions"] * panel["mean_charge_h"]
                                    / panel["capacity"]).clip(0, 1)
    panel["occupancy_density"] = (panel["sessions"] / panel["capacity"]).clip(lower=0)

    # calendar features
    hb = panel["hour_bucket"]
    panel["hour"] = hb.dt.hour
    panel["dow"] = hb.dt.dayofweek
    panel["is_weekend"] = (panel["dow"] >= 5).astype(int)
    panel["month"] = hb.dt.month
    panel["is_peak"] = panel["hour"].isin(C.PEAK_HOURS).astype(int)

    # temporal lag / rolling features for the demand model (per site, ordered)
    panel = panel.sort_values(["site", "hour_bucket"]).reset_index(drop=True)
    for site, idx in panel.groupby("site").groups.items():
        sub = panel.loc[idx].sort_values("hour_bucket")
        panel.loc[sub.index, "lag1_sessions"] = sub["sessions"].shift(1)
        panel.loc[sub.index, "lag24_sessions"] = sub["sessions"].shift(24)
        panel.loc[sub.index, "roll3_sessions"] = sub["sessions"].rolling(3, min_periods=1).mean()
        panel.loc[sub.index, "lag1_util"] = sub["station_utilization"].shift(1)
    panel[["lag1_sessions", "lag24_sessions", "roll3_sessions", "lag1_util"]] = \
        panel[["lag1_sessions", "lag24_sessions", "roll3_sessions", "lag1_util"]].fillna(0)

    _log(f"hourly panel: {len(panel)} site-hours across {panel['site'].nunique()} sites")
    return panel


# --------------------------------------------------------------------------- #
# UrbanEV (ST-EVCDP) cross-dataset profile
# --------------------------------------------------------------------------- #
def load_urbanev_profile() -> pd.DataFrame | None:
    """Aggregate UrbanEV 5-min occupancy into a city-wide hourly demand profile."""
    occ_path = C.URBANEV_DIR / "occupancy.csv"
    time_path = C.URBANEV_DIR / "time.csv"
    if not occ_path.exists() or not time_path.exists():
        warnings.warn("UrbanEV dataset not found - skipping cross-dataset validation.")
        _log("UrbanEV not available; continuing with ACN only.")
        return None
    _log("loading UrbanEV occupancy/volume/price ...")
    occ = pd.read_csv(occ_path)
    tim = pd.read_csv(time_path)
    vol = pd.read_csv(C.URBANEV_DIR / "volume.csv")
    price = pd.read_csv(C.URBANEV_DIR / "price.csv")

    zone_cols = [c for c in occ.columns if c != "timestamp"]
    df = pd.DataFrame({
        "hour": tim["hour"].values,
        "dow": pd.to_datetime(dict(year=tim["year"], month=tim["month"], day=tim["day"]))
                  .dt.dayofweek.values,
        "total_occupancy": occ[zone_cols].sum(axis=1).values,
        "total_volume": vol[[c for c in vol.columns if c != "timestamp"]].sum(axis=1).values,
        "mean_price": price[[c for c in price.columns if c != "timestamp"]].mean(axis=1).values,
    })
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    n_piles = int(occ[zone_cols].max().sum())  # crude capacity proxy
    df["occupancy_rate"] = (df["total_occupancy"] / max(n_piles, 1)).clip(0, 1)
    _log(f"UrbanEV: {len(df)} 5-min records across {len(zone_cols)} zones")
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save: bool = True):
    sessions = load_acn_sessions()
    sessions = add_concurrency_features(sessions)
    panel = build_hourly_panel(sessions)
    urbanev = load_urbanev_profile()

    if save:
        scols = ["sessionID", "site", "siteID", "stationID", "spaceID",
                 "connectionTime_local", "disconnectTime_local", "doneChargingTime_local",
                 "kWhDelivered", "connection_duration_h", "charging_duration_h", "idle_h",
                 "charger_utilization", "energy_cost_per_kwh", "revenue_baseline",
                 "gross_margin_baseline", "queue_proxy", "occupancy_density",
                 "hour", "dow", "is_weekend", "month"]
        sessions[scols].to_csv(C.DATA_PROCESSED / "acn_sessions_clean.csv", index=False)
        panel.to_csv(C.DATA_PROCESSED / "acn_hourly_panel.csv", index=False)
        if urbanev is not None:
            urbanev.to_csv(C.DATA_PROCESSED / "urbanev_profile.csv", index=False)
        _log(f"saved processed data to {C.DATA_PROCESSED}")
    return sessions, panel, urbanev


if __name__ == "__main__":
    run()
