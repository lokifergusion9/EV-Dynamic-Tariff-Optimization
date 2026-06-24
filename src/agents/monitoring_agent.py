"""Monitoring & Learning Agent.

Closes the agentic loop: it scores each pricing policy against simulated live
outcomes (revenue, utilization, wait time, pricing efficiency) and then *learns*
better tariff parameters across evaluation episodes via gradient-free hill-climbing.

Demand response (documented assumption):
    demand_factor = 1 + PRICE_ELASTICITY * (multiplier - 1)
i.e. raising price above baseline shifts/suppresses some demand (elasticity < 0),
discounts attract extra off-peak sessions. This is a transparent elasticity proxy,
NOT a causal claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
from agents.tariff_agent import TariffAgent


def _elasticity_for(util: np.ndarray) -> np.ndarray:
    """Utilization-varying elasticity: inelastic when busy, elastic when idle."""
    util = np.clip(np.asarray(util, dtype=float), 0, 1)
    # linear blend between the elastic (low-util) and inelastic (high-util) ends
    w = util  # 0 at idle -> off-peak elasticity, 1 at full -> peak elasticity
    return C.OFFPEAK_ELASTICITY + w * (C.PEAK_ELASTICITY - C.OFFPEAK_ELASTICITY)


def simulate_episode(panel: pd.DataFrame, tariff: TariffAgent,
                     util_col: str, elasticity=None) -> dict:
    """Apply a tariff policy to the panel and compute realized KPIs vs baseline."""
    df = panel.copy()
    util = df[util_col].values
    mult = tariff.multiplier(util)
    df["multiplier"] = mult
    df["dynamic_tariff"] = mult * tariff.baseline

    # demand response to relative price (utilization-varying elasticity)
    e = _elasticity_for(util) if elasticity is None else np.full_like(util, elasticity, dtype=float)
    df["demand_factor"] = np.clip(1 + e * (mult - 1), 0.3, 2.2)
    df["adj_sessions"] = df["sessions"] * df["demand_factor"]
    energy_per_session = np.where(df["sessions"] > 0,
                                  df["energy_kwh"] / df["sessions"].replace(0, np.nan), 0)
    energy_per_session = np.nan_to_num(energy_per_session)
    df["adj_energy"] = df["adj_sessions"] * energy_per_session

    # revenue
    rev_base = (df["energy_kwh"] * tariff.baseline).sum()
    rev_dyn = (df["adj_energy"] * df["dynamic_tariff"]).sum()

    # utilization after demand shift
    df["adj_utilization"] = np.clip(util * df["demand_factor"], 0, 1)

    # congestion / wait proxy = utilization above the comfort threshold during
    # peak hours (queueing builds once a cluster gets busy). Surge cuts it.
    peak = df["is_peak"] == 1
    wait_base = np.maximum(0, df.loc[peak, util_col] - C.CONGESTION_UTIL).sum()
    wait_dyn = np.maximum(0, df.loc[peak, "adj_utilization"] - C.CONGESTION_UTIL).sum()

    # off-peak uplift: extra sessions where utilization < discount threshold
    off = df[util_col] < C.DISCOUNT_UTIL
    base_off_sessions = df.loc[off, "sessions"].sum()
    offpeak_uplift = (df.loc[off, "adj_sessions"] - df.loc[off, "sessions"]).sum()
    offpeak_uplift_pct = offpeak_uplift / max(base_off_sessions, 1e-9) * 100

    total_energy = max(df["adj_energy"].sum(), 1e-9)
    return {
        "revenue_baseline": float(rev_base),
        "revenue_dynamic": float(rev_dyn),
        "revenue_gain_pct": float((rev_dyn - rev_base) / max(rev_base, 1e-9) * 100),
        "util_baseline": float(df[util_col].mean()),
        "util_dynamic": float(df["adj_utilization"].mean()),
        "wait_baseline": float(wait_base),
        "wait_dynamic": float(wait_dyn),
        "wait_reduction_pct": float((wait_base - wait_dyn) / max(wait_base, 1e-9) * 100),
        "offpeak_uplift_sessions": float(offpeak_uplift),
        "offpeak_uplift_pct": float(offpeak_uplift_pct),
        "pricing_efficiency": float(rev_dyn / total_energy),   # revenue per kWh delivered
        "customer_response_rate": float(df["demand_factor"].mean()),
    }


@dataclass
class MonitoringAgent:
    """Evaluates and improves the tariff policy over episodes (feedback loop)."""
    elasticity: float | None = None        # None -> utilization-varying elasticity
    history: list = field(default_factory=list)

    def _objective(self, kpi: dict) -> float:
        """Balance revenue, congestion relief and off-peak uplift (normalized %).

        Revenue is the primary objective; a policy that loses revenue is heavily
        penalized so the learner stays in the revenue-positive region.
        """
        rev = kpi["revenue_gain_pct"]
        penalty = 3.0 * min(rev, 0.0)        # extra sting for revenue loss
        return (rev + penalty
                + 0.4 * kpi["wait_reduction_pct"]
                + 0.3 * kpi["offpeak_uplift_pct"])

    def learn(self, panel: pd.DataFrame, util_col: str,
              episodes: int = 25, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
        """Hill-climb tariff parameters; log KPIs per episode to show improvement."""
        rng = np.random.default_rng(seed)
        # Start from a near-flat (barely dynamic) policy so the feedback loop has
        # room to visibly improve its decisions over episodes.
        agent = TariffAgent(max_mult=1.10, min_mult=0.95,
                            surge_util=C.SURGE_UTIL, discount_util=C.DISCOUNT_UTIL)
        best_kpi = simulate_episode(panel, agent, util_col, self.elasticity)
        best_score = self._objective(best_kpi)
        self.history = [{"episode": 0, "max_mult": agent.max_mult,
                         "min_mult": agent.min_mult, "surge_util": agent.surge_util,
                         "score": best_score, **best_kpi}]

        for ep in range(1, episodes + 1):
            cand = TariffAgent(
                max_mult=float(np.clip(agent.max_mult + rng.normal(0, 0.08), 1.05, 2.0)),
                min_mult=float(np.clip(agent.min_mult + rng.normal(0, 0.05), 0.5, 0.95)),
                surge_util=float(np.clip(agent.surge_util + rng.normal(0, 0.03), 0.6, 0.9)),
                discount_util=float(np.clip(agent.discount_util + rng.normal(0, 0.03), 0.15, 0.45)),
            )
            kpi = simulate_episode(panel, cand, util_col, self.elasticity)
            score = self._objective(kpi)
            if score > best_score:                  # accept improving move
                agent, best_score, best_kpi = cand, score, kpi
            self.history.append({"episode": ep, "max_mult": agent.max_mult,
                                 "min_mult": agent.min_mult, "surge_util": agent.surge_util,
                                 "score": best_score, **best_kpi})

        self.best_agent = agent
        self.best_kpi = best_kpi
        return pd.DataFrame(self.history)
