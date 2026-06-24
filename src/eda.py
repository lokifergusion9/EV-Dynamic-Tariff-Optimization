"""Stage 2 - Exploratory Data Analysis.

Generates insight-driven, well-labeled figures saved to outputs/figures/.
Every chart carries a short pricing implication in its title/annotation so the
EDA ties directly to the dynamic-tariff objective (OP'26 brief).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C

sns.set_theme(style="whitegrid", palette="viridis")
PEAK_COLOR, OFF_COLOR = "#c0392b", "#27ae60"


def _save(fig, name):
    path = C.OUT_FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] wrote {path.name}")
    return path


def intraday_demand(panel: pd.DataFrame):
    """Mean sessions by hour-of-day -> reveals the peak window to surge-price."""
    prof = panel.groupby("hour")["sessions"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(prof.index, prof.values,
                  color=[PEAK_COLOR if h in C.PEAK_HOURS else "#5b8db8" for h in prof.index])
    ax.set(xlabel="Hour of day (site-local)", ylabel="Mean sessions / hour",
           title="Intraday charging demand — daytime workplace peak drives congestion pricing")
    ax.axvspan(min(C.PEAK_HOURS) - 0.5, max(C.PEAK_HOURS) + 0.5, color=PEAK_COLOR, alpha=0.06)
    ax.annotate("Peak window → surge candidate", (12, prof.max()),
                ha="center", va="bottom", color=PEAK_COLOR, fontweight="bold")
    return _save(fig, "01_intraday_demand.png")


def weekday_weekend(panel: pd.DataFrame):
    prof = panel.groupby(["is_weekend", "hour"])["sessions"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if 0 in prof: ax.plot(prof.index, prof[0], "-o", ms=3, label="Weekday", color="#2c3e50")
    if 1 in prof: ax.plot(prof.index, prof[1], "-o", ms=3, label="Weekend", color=OFF_COLOR)
    ax.set(xlabel="Hour of day", ylabel="Mean sessions / hour",
           title="Weekday vs weekend — weekends are off-peak, ideal for discount uplift")
    ax.legend()
    return _save(fig, "02_weekday_weekend.png")


def utilization_heatmap(panel: pd.DataFrame):
    piv = (panel.pivot_table(index="dow", columns="hour",
                             values="station_utilization", aggfunc="mean")
                .reindex(index=range(7)))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.heatmap(piv, cmap="rocket_r", ax=ax, cbar_kws={"label": "Station utilization"},
                vmin=0, vmax=min(1, np.nanpercentile(piv.values, 99)))
    ax.set(xlabel="Hour of day", ylabel="Day of week (0=Mon)",
           title="Utilization heatmap — bright cells = surge zones, dark = discount zones")
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    return _save(fig, "03_utilization_heatmap.png")


def session_distributions(sessions: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sns.histplot(sessions["kWhDelivered"].clip(upper=sessions["kWhDelivered"].quantile(0.99)),
                 bins=40, ax=axes[0], color="#2980b9")
    axes[0].set(title="Energy delivered / session", xlabel="kWh")
    sns.histplot(sessions["connection_duration_h"].clip(upper=24), bins=40, ax=axes[1], color="#8e44ad")
    axes[1].set(title="Connection (plugged-in) duration", xlabel="hours")
    sns.histplot(sessions["charger_utilization"], bins=30, ax=axes[2], color="#16a085")
    axes[2].axvline(sessions["charger_utilization"].mean(), color="red", ls="--",
                    label=f"mean={sessions['charger_utilization'].mean():.2f}")
    axes[2].set(title="Charger utilization (charge/connect)", xlabel="ratio"); axes[2].legend()
    fig.suptitle("Session-level distributions — long idle tails reveal under-used capacity to monetize",
                 fontweight="bold")
    return _save(fig, "04_session_distributions.png")


def volatility_by_period(panel: pd.DataFrame):
    """Coefficient of variation of demand across peak / shoulder / off-peak."""
    def period(h):
        if h in C.PEAK_HOURS: return "Peak"
        if h in C.OFFPEAK_HOURS: return "Off-peak"
        return "Shoulder"
    panel = panel.copy()
    panel["period"] = panel["hour"].map(period)
    stat = panel.groupby("period")["sessions"].agg(["mean", "std"])
    stat["cv"] = stat["std"] / stat["mean"].replace(0, np.nan)
    stat = stat.reindex(["Off-peak", "Shoulder", "Peak"])
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bars = ax.bar(stat.index, stat["cv"], color=[OFF_COLOR, "#f39c12", PEAK_COLOR])
    ax.bar_label(bars, fmt="%.2f")
    ax.set(ylabel="Coefficient of variation (demand)",
           title="Demand volatility by period — higher CV means more pricing headroom")
    return _save(fig, "05_volatility_by_period.png")


def occupancy_vs_utilization(panel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    s = panel.sample(min(len(panel), 4000), random_state=C.RANDOM_STATE)
    sc = ax.scatter(s["occupancy_density"], s["station_utilization"],
                    c=s["hour"], cmap="twilight", s=14, alpha=0.6)
    ax.axhline(C.SURGE_UTIL, color=PEAK_COLOR, ls="--", label=f"surge @ {C.SURGE_UTIL:.0%}")
    ax.axhline(C.DISCOUNT_UTIL, color=OFF_COLOR, ls="--", label=f"discount @ {C.DISCOUNT_UTIL:.0%}")
    ax.set(xlabel="Occupancy density (sessions / station)", ylabel="Station utilization",
           title="Occupancy vs utilization — pricing bands overlaid")
    fig.colorbar(sc, ax=ax, label="hour of day"); ax.legend()
    return _save(fig, "06_occupancy_vs_utilization.png")


def acn_vs_urbanev(panel: pd.DataFrame, urbanev: pd.DataFrame | None):
    if urbanev is None:
        return None
    acn = panel.groupby("hour")["sessions"].mean()
    acn = acn / acn.max()
    ue = urbanev.groupby("hour")["occupancy_rate"].mean()
    ue = ue / ue.max()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(acn.index, acn.values, "-o", ms=3, label="ACN (Caltech, workplace)", color="#2c3e50")
    ax.plot(ue.index, ue.values, "-s", ms=3, label="UrbanEV (Shenzhen, public)", color="#e67e22")
    ax.set(xlabel="Hour of day", ylabel="Normalized demand (0-1)",
           title="Cross-dataset intraday demand — workplace daytime peak vs urban evening peak")
    ax.legend()
    return _save(fig, "07_acn_vs_urbanev.png")


def run(sessions: pd.DataFrame, panel: pd.DataFrame, urbanev: pd.DataFrame | None):
    print("[eda] generating figures ...")
    intraday_demand(panel)
    weekday_weekend(panel)
    utilization_heatmap(panel)
    session_distributions(sessions)
    volatility_by_period(panel)
    occupancy_vs_utilization(panel)
    acn_vs_urbanev(panel, urbanev)
    print(f"[eda] all figures in {C.OUT_FIG}")


if __name__ == "__main__":
    import preprocessing as P
    s, p, u = P.run(save=False)
    run(s, p, u)
