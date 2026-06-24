"""Stage 4 - Evaluation & reporting.

Collects the OP'26 evaluation metrics for all three agents into tidy CSVs and a
few summary figures:
  Demand agent     : RMSE, MAE, R2 (load & utilization), congestion accuracy/AUC
  Tariff agent     : Revenue Gain %, utilization before/after, off-peak uplift
  Monitoring agent : wait-time reduction, customer response rate, pricing efficiency
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C


def save_demand_metrics(metrics: dict, importance: pd.DataFrame) -> pd.DataFrame:
    rows = [{"agent": "DemandPrediction", "metric": k, "value": round(v, 4)}
            for k, v in metrics.items()]
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_CSV / "demand_metrics.csv", index=False)
    importance.to_csv(C.OUT_CSV / "demand_feature_importance.csv", index=False)
    print("[eval] demand metrics ->", df["value"].tolist())
    return df


def save_tariff_schedule(schedule: pd.DataFrame, recommendations: pd.DataFrame):
    schedule.to_csv(C.OUT_CSV / "tariff_schedule.csv", index=False)
    cols = ["hour_bucket", "site", "pred_utilization", "price_multiplier",
            "dynamic_tariff", "baseline_tariff", "pricing_action"]
    cols = [c for c in cols if c in recommendations.columns]
    recommendations[cols].to_csv(C.OUT_CSV / "tariff_recommendations.csv", index=False)
    print("[eval] tariff schedule + recommendations saved")


def save_pricing_outcomes(baseline_kpi: dict, best_kpi: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "metric": ["revenue_gain_pct", "util_baseline", "util_dynamic",
                   "wait_reduction_pct", "offpeak_uplift_sessions",
                   "pricing_efficiency", "customer_response_rate"],
        "default_policy": [baseline_kpi[k] for k in
                           ["revenue_gain_pct", "util_baseline", "util_dynamic",
                            "wait_reduction_pct", "offpeak_uplift_sessions",
                            "pricing_efficiency", "customer_response_rate"]],
        "learned_policy": [best_kpi[k] for k in
                           ["revenue_gain_pct", "util_baseline", "util_dynamic",
                            "wait_reduction_pct", "offpeak_uplift_sessions",
                            "pricing_efficiency", "customer_response_rate"]],
    }).round(3)
    df.to_csv(C.OUT_CSV / "pricing_outcomes.csv", index=False)
    print("[eval] pricing outcomes saved")
    return df


def save_learning_curve(history: pd.DataFrame):
    history.to_csv(C.OUT_CSV / "monitoring_episodes.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(history["episode"], history["score"], "-o", ms=3, color="#16a085")
    ax2 = ax[0].twinx()
    ax2.plot(history["episode"], history["pricing_efficiency"], "--", color="#7f8c8d",
             alpha=0.8, label="pricing efficiency")
    ax2.set_ylabel("Pricing efficiency (INR/kWh)", color="#7f8c8d")
    ax[0].set(xlabel="Evaluation episode", ylabel="Composite policy score",
              title="Feedback loop improves pricing decisions over episodes")
    ax[1].plot(history["episode"], history["revenue_gain_pct"], "-o", ms=3,
               color="#2980b9", label="Revenue gain %")
    ax[1].plot(history["episode"], history["wait_reduction_pct"], "-s", ms=3,
               color="#c0392b", label="Wait reduction %")
    ax[1].set(xlabel="Evaluation episode", ylabel="%",
              title="Learning agent: revenue vs congestion relief")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(C.OUT_FIG / "08_learning_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[eval] learning curve saved")


def tariff_curve_figure(schedule: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(schedule["utilization"], schedule["tariff_inr_per_kwh"], "-o", ms=3, color="#8e44ad")
    ax.axhline(C.BASELINE_TARIFF, ls="--", color="grey", label=f"baseline INR {C.BASELINE_TARIFF}/kWh")
    ax.axvspan(C.SURGE_UTIL, 1.0, color="#c0392b", alpha=0.08, label="surge zone")
    ax.axvspan(0, C.DISCOUNT_UTIL, color="#27ae60", alpha=0.08, label="discount zone")
    ax.set(xlabel="Predicted utilization", ylabel="Tariff (INR / kWh)",
           title="Dynamic tariff curve — surge above 80%, discount below 30% utilization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.OUT_FIG / "09_tariff_curve.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[eval] tariff curve saved")


def write_summary(demand_metrics: dict, best_kpi: dict):
    lines = [
        "OP'26 - Agentic Tariff Optimization: Results Summary",
        "=" * 55,
        "",
        "Demand Prediction Agent:",
        f"  Charging-load   RMSE={demand_metrics['load_RMSE']:.3f}  "
        f"MAE={demand_metrics['load_MAE']:.3f}  R2={demand_metrics['load_R2']:.3f}",
        f"  Utilization     RMSE={demand_metrics['util_RMSE']:.3f}  "
        f"MAE={demand_metrics['util_MAE']:.3f}  R2={demand_metrics['util_R2']:.3f}",
    ]
    if "congestion_ROC_AUC" in demand_metrics:
        lines.append(f"  Congestion      ACC={demand_metrics['congestion_accuracy']:.3f}  "
                     f"ROC-AUC={demand_metrics['congestion_ROC_AUC']:.3f}")
    lines += [
        "",
        "Tariff + Monitoring/Learning Agent (best learned policy vs INR 15/kWh):",
        f"  Revenue Gain %        : {best_kpi['revenue_gain_pct']:+.2f}%",
        f"  Avg wait reduction    : {best_kpi['wait_reduction_pct']:+.2f}%",
        f"  Off-peak uplift       : {best_kpi['offpeak_uplift_sessions']:+.1f} sessions",
        f"  Pricing efficiency    : INR {best_kpi['pricing_efficiency']:.2f} / kWh",
        f"  Customer response rate: {best_kpi['customer_response_rate']:.3f} (demand factor)",
    ]
    text = "\n".join(lines)
    (C.OUT_CSV.parent / "RESULTS_SUMMARY.txt").write_text(text, encoding="utf-8")
    print("\n" + text + "\n")
