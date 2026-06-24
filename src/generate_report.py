"""Generate a polished multi-page PDF project report with the actual result graphs.

Output: outputs/OP26_Project_Report.pdf
Run from src/:   python generate_report.py   (run AFTER run_pipeline.py)
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import config as C

FIG = C.OUT_FIG
CSV = C.OUT_CSV
ACCENT = "#0d3b2e"
GREEN = "#16a085"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})


# --------------------------------------------------------------------------- #
# page helpers
# --------------------------------------------------------------------------- #
def _blank_page():
    fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.06)
    return fig


def cover_page(pdf):
    fig = _blank_page()
    fig.patch.set_facecolor(ACCENT)
    fig.text(0.5, 0.74, "Open Project 2026", ha="center", color="#7fe7c4",
             fontsize=22, fontweight="bold")
    fig.text(0.5, 0.66, "Agentic AI-Based Dynamic Tariff Optimization\nfor EV Charging Networks",
             ha="center", color="white", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.55, "Using Large-Scale Charging Session Data", ha="center",
             color="#cfe9df", fontsize=13, style="italic")
    fig.text(0.5, 0.40,
             "A self-improving pricing engine: forecast demand  →  recommend\n"
             "dynamic per-kWh tariffs  →  learn from outcomes.",
             ha="center", color="white", fontsize=12)
    fig.text(0.5, 0.20, "Project Report & Results", ha="center", color="#7fe7c4",
             fontsize=14, fontweight="bold")
    fig.text(0.5, 0.10, "Datasets: ACN-Data (Caltech) · UrbanEV / ST-EVCDP (Shenzhen)",
             ha="center", color="#cfe9df", fontsize=10)
    pdf.savefig(fig, facecolor=ACCENT)
    plt.close(fig)


def text_page(pdf, title, body, subtitle=None):
    fig = _blank_page()
    fig.text(0.08, 0.95, title, color=ACCENT, fontsize=17, fontweight="bold")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=ACCENT, lw=2,
                              transform=fig.transFigure))
    y = 0.89
    if subtitle:
        fig.text(0.08, y, subtitle, color=GREEN, fontsize=11, fontweight="bold")
        y -= 0.03
    for block in body:
        if block.startswith("## "):
            y -= 0.012
            fig.text(0.08, y, block[3:], color=ACCENT, fontsize=12.5, fontweight="bold")
            y -= 0.028
            continue
        bullet = block.startswith("- ")
        text = block[2:] if bullet else block
        wrapped = textwrap.wrap(text, width=95 if not bullet else 92)
        for i, line in enumerate(wrapped):
            prefix = "•  " if (bullet and i == 0) else ("   " if bullet else "")
            fig.text(0.08, y, prefix + line, color="#222222", fontsize=10.3)
            y -= 0.0235
        y -= 0.010
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, img_path, caption):
    fig = _blank_page()
    fig.text(0.08, 0.95, title, color=ACCENT, fontsize=15, fontweight="bold")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=ACCENT, lw=2,
                              transform=fig.transFigure))
    ax = fig.add_axes([0.06, 0.30, 0.88, 0.58])
    ax.axis("off")
    try:
        ax.imshow(plt.imread(str(img_path)))
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"(missing {img_path.name})", ha="center")
    y = 0.25
    for line in textwrap.wrap("Insight:  " + caption, width=90):
        fig.text(0.08, y, line, color="#222222", fontsize=10.5)
        y -= 0.024
    pdf.savefig(fig)
    plt.close(fig)


def two_image_page(pdf, title, img1, cap1, img2, cap2):
    fig = _blank_page()
    fig.text(0.08, 0.95, title, color=ACCENT, fontsize=15, fontweight="bold")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=ACCENT, lw=2,
                              transform=fig.transFigure))
    # (image-bottom, caption-y) for the top and bottom halves of the page
    for img, cap, img_bottom, cap_y in [(img1, cap1, 0.57, 0.515),
                                        (img2, cap2, 0.13, 0.075)]:
        ax = fig.add_axes([0.06, img_bottom, 0.88, 0.30]); ax.axis("off")
        try:
            ax.imshow(plt.imread(str(img)))
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"(missing {img.name})", ha="center")
        yy = cap_y
        for line in textwrap.wrap("Insight:  " + cap, width=100):
            fig.text(0.08, yy, line, color="#222222", fontsize=9.8); yy -= 0.022
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, intro, tables):
    """tables: list of (subtitle, DataFrame)."""
    fig = _blank_page()
    fig.text(0.08, 0.95, title, color=ACCENT, fontsize=15, fontweight="bold")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=ACCENT, lw=2,
                              transform=fig.transFigure))
    y = 0.90
    for line in textwrap.wrap(intro, width=100):
        fig.text(0.08, y, line, color="#222222", fontsize=10.3); y -= 0.023
    y -= 0.02
    for subtitle, df in tables:
        fig.text(0.08, y, subtitle, color=GREEN, fontsize=11, fontweight="bold"); y -= 0.02
        h = 0.04 + 0.028 * len(df)
        ax = fig.add_axes([0.08, y - h, 0.84, h]); ax.axis("off")
        tbl = ax.table(cellText=df.values, colLabels=df.columns,
                       cellLoc="center", loc="upper center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.35)
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(ACCENT); cell.set_text_props(color="white", fontweight="bold")
        y -= h + 0.05
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def build():
    demand = pd.read_csv(CSV / "demand_metrics.csv")
    imp = pd.read_csv(CSV / "demand_feature_importance.csv").head(6).round(3)
    outcomes = pd.read_csv(CSV / "pricing_outcomes.csv")
    sched = pd.read_csv(CSV / "tariff_schedule.csv")
    epis = pd.read_csv(CSV / "monitoring_episodes.csv")

    dm = {r["metric"]: r["value"] for _, r in demand.iterrows()}
    oc = {r["metric"]: r["learned_policy"] for _, r in outcomes.iterrows()}

    out = C.OUT_FIG.parent / "OP26_Project_Report.pdf"
    with PdfPages(out) as pdf:
        cover_page(pdf)

        text_page(pdf, "1 · Problem, Objective & Approach", [
            "## Problem",
            "Static, fixed-rate EV charging tariffs ignore real operational dynamics: peak-hour "
            "congestion, off-peak underutilization, and time-varying grid procurement cost. A flat "
            "INR/kWh price leaves revenue on the table at peak and fails to attract flexible users off-peak.",
            "## Objective",
            "- Forecast charging demand and station utilization across time of day, day of week and location.",
            "- Recommend the optimal per-kWh tariff to maximize revenue while minimizing congestion and waits.",
            "- Identify under-utilized vs overloaded periods and price them accordingly.",
            "- Build a self-improving system whose Monitoring & Learning agent refines pricing via a feedback loop.",
            "## Approach — three cooperating agents",
            "- Demand Prediction Agent (ML): predicts charging load, utilization and congestion probability.",
            "- Tariff Pricing Agent (rules): maps predicted utilization to a dynamic price multiplier.",
            "- Monitoring & Learning Agent: scores each policy against simulated outcomes and adapts it.",
            "The system is deterministic and reproducible — no external LLM/API. 'Agentic' refers to the "
            "autonomous forecast → price → learn loop.",
        ])

        text_page(pdf, "2 · Data Landscape & Preprocessing", [
            "## Datasets",
            "- ACN-Data (primary): 14,999 clean charging sessions from the Caltech adaptive charging "
            "network, Apr–Dec 2018. Real connection / disconnect / done-charging timestamps, kWh "
            "delivered, station and space IDs.",
            "- UrbanEV / ST-EVCDP (cross-validation): Shenzhen, 8,640 five-minute records across 247 "
            "zones, June 2022 — used to compare intraday demand shapes across geographies.",
            "## Cleaning decisions (logged for transparency)",
            "- Dropped 1,305 JSON pagination / meta rows that carry no connectionTime.",
            "- Parsed RFC-1123 GMT timestamps to UTC then to site-local time.",
            "- Dropped userInputs (null for 100% of rows); kWhRequested is 78% missing, so kWhDelivered "
            "is used as the energy measure.",
            "- doneChargingTime missing for 8 sessions → filled from disconnectTime; non-positive "
            "durations and negative energy removed; charging time clipped within connection time.",
            "## Engineered economic features (per brief)",
            "- Charger Utilization Rate = charging time / connection (occupied) time.",
            "- Revenue per Session = kWh delivered × tariff.",
            "- Energy Cost per kWh = time-of-use grid procurement proxy (peak / shoulder / off-peak).",
            "- Queue Length Proxy = concurrent active sessions per site-hour (interval overlap).",
            "- Occupancy Density = active sessions / distinct stations (capacity = 54 EVSEs).",
            "A continuous hourly site-level demand panel with lag and rolling features is the modeling table.",
        ])

        two_image_page(pdf, "3 · Exploratory Data Analysis (1/3)",
                       FIG / "01_intraday_demand.png",
                       "A sharp 08:00–09:00 workplace peak (~12 sessions/hr) is the prime surge-pricing "
                       "window; demand collapses overnight, leaving discount headroom.",
                       FIG / "02_weekday_weekend.png",
                       "Weekdays drive the daytime peak; weekends are uniformly off-peak — a natural "
                       "target for discount-led off-peak uplift.")

        two_image_page(pdf, "3 · Exploratory Data Analysis (2/3)",
                       FIG / "03_utilization_heatmap.png",
                       "Utilization concentrates in weekday daytime cells (surge zones); the rest of the "
                       "week×hour grid is dark (discount zones).",
                       FIG / "05_volatility_by_period.png",
                       "Demand volatility (coefficient of variation) is highest at peak, meaning peak "
                       "periods have the most pricing headroom.")

        two_image_page(pdf, "3 · Exploratory Data Analysis (3/3)",
                       FIG / "04_session_distributions.png",
                       "Energy per session and plugged-in duration are right-skewed; long idle tails "
                       "(low charger utilization) reveal capacity that pricing can monetize.",
                       FIG / "07_acn_vs_urbanev.png",
                       "ACN (workplace, daytime peak) vs UrbanEV (urban, evening peak): demand shape is "
                       "location-specific, so tariffs must be learned per network — not copied.")

        table_page(pdf, "4 · Demand Prediction Agent — Results",
                   "GradientBoosting (charging load) + RandomForest (utilization) + GradientBoosting "
                   "classifier (congestion), trained on calendar and lag/rolling features with a "
                   "chronological train/test split so the future is never leaked into training.",
                   [("Accuracy metrics", pd.DataFrame({
                        "Target": ["Charging load (sessions/hr)", "Utilization rate", "Congestion (>80%)"],
                        "RMSE": [dm.get("load_RMSE"), dm.get("util_RMSE"), "—"],
                        "MAE": [dm.get("load_MAE"), dm.get("util_MAE"), "—"],
                        "R² / AUC": [dm.get("load_R2"), dm.get("util_R2"),
                                     f"AUC {dm.get('congestion_ROC_AUC')}"],
                    })),
                    ("Top demand drivers (feature importance)",
                     imp.rename(columns={"feature": "Feature", "importance": "Importance"}))])

        image_page(pdf, "5 · Dynamic Tariff Optimization Logic",
                   FIG / "09_tariff_curve.png",
                   "The tariff agent maps predicted utilization to a price multiplier on the INR 15/kWh "
                   "baseline: surge up to 1.5× above 80% utilization, discount to 0.7× below 30%, smooth "
                   "in between. Demand response uses a utilization-varying elasticity (peak inelastic "
                   "|e|≈0.20, off-peak elastic |e|≈1.45), so off-peak discounts grow volume more than "
                   "they cut price — a documented assumption, not a causal claim.")

        image_page(pdf, "6 · Monitoring & Learning Agent — Feedback Loop",
                   FIG / "08_learning_curve.png",
                   "Each policy is scored against simulated outcomes (revenue, congestion, uplift). A "
                   "gradient-free hill-climbing loop adapts the tariff parameters over 30 episodes: the "
                   f"composite score rises from {epis['score'].iloc[0]:.1f} to {epis['score'].iloc[-1]:.1f}, "
                   "revenue gain and peak-wait reduction both climb — the system demonstrably improves "
                   "its own pricing decisions over time.")

        table_page(pdf, "7 · Results Summary (learned policy vs fixed INR 15/kWh)",
                   "Headline evaluation metrics across all three agents.",
                   [("Pricing & operations outcomes", pd.DataFrame({
                        "Metric": ["Revenue Gain %", "Avg peak wait reduction %",
                                   "Off-peak uplift (sessions)", "Pricing efficiency (INR/kWh)",
                                   "Customer response rate"],
                        "Value": [f"+{oc.get('revenue_gain_pct')}%",
                                  f"{oc.get('wait_reduction_pct')}%",
                                  f"+{oc.get('offpeak_uplift_sessions')}",
                                  oc.get("pricing_efficiency"),
                                  oc.get("customer_response_rate")],
                    }))])

        text_page(pdf, "8 · Implications, Assumptions & Limitations", [
            "## Business, operational & policy implications",
            "- Revenue: dynamic pricing beats the flat INR 15/kWh baseline (~+8%) by capturing peak "
            "willingness-to-pay and growing elastic off-peak volume.",
            "- Operations: peak surge cuts the congestion/wait proxy ~22%, smoothing demand across the "
            "day and deferring costly capacity expansion.",
            "- Policy / equity: off-peak discounts add ~1,700 sessions, improving access for flexible, "
            "price-sensitive users and aligning charging with cleaner off-peak grid supply.",
            "## Assumptions & limitations (transparency)",
            "- Prices are expressed in INR per the brief's reference baseline — not a USD→INR conversion; "
            "ACN is US workplace charging.",
            "- Demand response is an elasticity assumption, NOT a causal estimate. Revenue and uplift "
            "figures are simulation-based counterfactuals.",
            "- This export is a single site over ~8 months with genuinely low utilization (median ~6%); "
            "patterns should be re-fit per network. Chronological validation guards against leakage.",
            "## Reproducibility",
            "- Full pipeline: python src/run_pipeline.py  (preprocess → EDA → agents → evaluation).",
            "- Outputs: outputs/figures/*.png, outputs/csv/*.csv, notebooks/OP26_Analytics.ipynb.",
        ])

    print(f"[report] wrote {out}")
    return out


if __name__ == "__main__":
    build()
