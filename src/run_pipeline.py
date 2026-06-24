"""End-to-end orchestrator for the OP'26 Agentic Tariff Optimization pipeline.

Run from the src/ directory:   python run_pipeline.py
Stages: preprocess -> EDA -> demand agent -> tariff agent -> monitoring/learning
        -> evaluation (CSVs, figures, summary).
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import preprocessing as P
import eda
import evaluation as E
from agents.demand_agent import DemandAgent
from agents.tariff_agent import TariffAgent
from agents.monitoring_agent import MonitoringAgent, simulate_episode


def main():
    print("\n=== STAGE 1: PREPROCESSING ===")
    sessions, panel, urbanev = P.run(save=True)

    print("\n=== STAGE 2: EDA ===")
    eda.run(sessions, panel, urbanev)

    print("\n=== STAGE 3a: DEMAND PREDICTION AGENT ===")
    demand = DemandAgent().fit(panel)
    print("[demand] metrics:", {k: round(v, 3) for k, v in demand.metrics.items()})
    forecast = demand.predict(panel)

    print("\n=== STAGE 3b: TARIFF PRICING AGENT ===")
    tariff = TariffAgent()
    recommendations = tariff.recommend(forecast, util_col="pred_utilization")
    schedule = tariff.schedule()
    print("[tariff] action mix:", recommendations["pricing_action"].str[:6].value_counts().to_dict())

    print("\n=== STAGE 3c: MONITORING & LEARNING AGENT ===")
    monitor = MonitoringAgent()
    baseline_kpi = simulate_episode(panel, TariffAgent(), util_col="station_utilization")
    history = monitor.learn(panel, util_col="station_utilization", episodes=30)
    print(f"[monitor] revenue gain {baseline_kpi['revenue_gain_pct']:.2f}% "
          f"-> {monitor.best_kpi['revenue_gain_pct']:.2f}% after learning")

    print("\n=== STAGE 4: EVALUATION ===")
    E.save_demand_metrics(demand.metrics, demand.feature_importance())
    E.save_tariff_schedule(schedule, recommendations)
    E.tariff_curve_figure(schedule)
    E.save_pricing_outcomes(baseline_kpi, monitor.best_kpi)
    E.save_learning_curve(history)
    E.write_summary(demand.metrics, monitor.best_kpi)
    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
