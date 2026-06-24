"""Tariff Pricing Agent.

Translates the demand agent's utilization forecast into a dynamic per-kWh tariff.

Rule (transparent, monotonic in utilization):
  util >= SURGE_UTIL (0.80)    -> surge,    multiplier ramps up to MAX_MULTIPLIER
  util <= DISCOUNT_UTIL (0.30) -> discount, multiplier ramps down to MIN_MULTIPLIER
  in between                    -> linear interpolation around 1.0x baseline

The multiplier is applied to the BASELINE_TARIFF (INR 15/kWh). Parameters
(max/min multiplier, thresholds, elasticity) are tunable and are exactly what the
Monitoring & Learning Agent adapts over evaluation episodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C


@dataclass
class TariffAgent:
    baseline: float = C.BASELINE_TARIFF
    surge_util: float = C.SURGE_UTIL
    discount_util: float = C.DISCOUNT_UTIL
    max_mult: float = C.MAX_MULTIPLIER
    min_mult: float = C.MIN_MULTIPLIER

    # ------------------------------------------------------------------ #
    def multiplier(self, util):
        """Vectorized utilization -> price multiplier (piecewise linear)."""
        util = np.asarray(util, dtype=float)
        m = np.ones_like(util)

        # surge region: util in [surge_util, 1] -> 1.0 .. max_mult
        surge = util >= self.surge_util
        span = max(1.0 - self.surge_util, 1e-6)
        m = np.where(surge,
                     1.0 + (util - self.surge_util) / span * (self.max_mult - 1.0), m)

        # discount region: util in [0, discount_util] -> min_mult .. 1.0
        disc = util <= self.discount_util
        dspan = max(self.discount_util, 1e-6)
        m = np.where(disc,
                     self.min_mult + (util / dspan) * (1.0 - self.min_mult), m)

        return np.clip(m, self.min_mult, self.max_mult)

    def price(self, util):
        return self.multiplier(util) * self.baseline

    # ------------------------------------------------------------------ #
    def recommend(self, forecast: pd.DataFrame, util_col="pred_utilization") -> pd.DataFrame:
        """Attach dynamic tariff + a human-readable action to each forecast row."""
        out = forecast.copy()
        out["price_multiplier"] = self.multiplier(out[util_col])
        out["dynamic_tariff"] = out["price_multiplier"] * self.baseline
        out["baseline_tariff"] = self.baseline

        def action(u):
            if u >= self.surge_util:
                return "SURGE (high demand - raise price, relieve congestion)"
            if u <= self.discount_util:
                return "DISCOUNT (low demand - attract off-peak sessions)"
            return "HOLD (balanced demand)"
        out["pricing_action"] = out[util_col].map(action)
        return out

    def schedule(self) -> pd.DataFrame:
        """A canonical utilization -> tariff lookup table (for the deck/CSV)."""
        u = np.round(np.arange(0, 1.001, 0.05), 2)
        return pd.DataFrame({
            "utilization": u,
            "multiplier": np.round(self.multiplier(u), 3),
            "tariff_inr_per_kwh": np.round(self.price(u), 2),
        })
