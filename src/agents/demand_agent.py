"""Demand Prediction Agent.

Forecasts, per site-hour, three quantities the pricing agent consumes:
  * expected charging load   (number of sessions)        -> regression
  * station utilization rate                              -> regression
  * congestion probability   P(utilization > SURGE_UTIL) -> classification

Uses gradient-boosted / random-forest models on calendar + lag features with a
chronological (time-ordered) train/test split so we never leak the future.
Reports RMSE / MAE / R2 (regression) and accuracy / ROC-AUC (congestion).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

import config as C

FEATURES = ["hour", "dow", "is_weekend", "month", "is_peak",
            "lag1_sessions", "lag24_sessions", "roll3_sessions", "lag1_util",
            "capacity", "mean_energy_cost"]


@dataclass
class DemandAgent:
    """Trains demand models and emits forecasts the tariff agent acts on."""
    random_state: int = C.RANDOM_STATE
    models: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    test_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    def _split(self, panel: pd.DataFrame, test_frac=0.2):
        panel = panel.sort_values("hour_bucket").reset_index(drop=True)
        cut = int(len(panel) * (1 - test_frac))
        return panel.iloc[:cut].copy(), panel.iloc[cut:].copy()

    def fit(self, panel: pd.DataFrame) -> "DemandAgent":
        train, test = self._split(panel)
        Xtr, Xte = train[FEATURES], test[FEATURES]

        # 1) charging load (sessions/hour)
        load = GradientBoostingRegressor(random_state=self.random_state)
        load.fit(Xtr, train["sessions"])
        # 2) utilization rate
        util = RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1)
        util.fit(Xtr, train["station_utilization"])
        # 3) congestion classifier
        ytr_c = (train["station_utilization"] > C.SURGE_UTIL).astype(int)
        yte_c = (test["station_utilization"] > C.SURGE_UTIL).astype(int)
        if ytr_c.nunique() < 2:           # fallback if no congested hours in train
            cong = None
        else:
            cong = GradientBoostingClassifier(random_state=self.random_state)
            cong.fit(Xtr, ytr_c)

        self.models = {"load": load, "util": util, "congestion": cong}

        # ---- metrics on the held-out (future) slice ---------------------- #
        pred_load = load.predict(Xte)
        pred_util = np.clip(util.predict(Xte), 0, 1)
        self.metrics = {
            "load_RMSE": float(np.sqrt(mean_squared_error(test["sessions"], pred_load))),
            "load_MAE": float(mean_absolute_error(test["sessions"], pred_load)),
            "load_R2": float(r2_score(test["sessions"], pred_load)),
            "util_RMSE": float(np.sqrt(mean_squared_error(test["station_utilization"], pred_util))),
            "util_MAE": float(mean_absolute_error(test["station_utilization"], pred_util)),
            "util_R2": float(r2_score(test["station_utilization"], pred_util)),
        }
        if cong is not None:
            proba = cong.predict_proba(Xte)[:, 1]
            self.metrics["congestion_accuracy"] = float(accuracy_score(yte_c, proba > 0.5))
            self.metrics["congestion_ROC_AUC"] = float(roc_auc_score(yte_c, proba)) \
                if yte_c.nunique() > 1 else float("nan")

        test = test.assign(pred_sessions=pred_load,
                           pred_utilization=pred_util,
                           pred_congestion_prob=(cong.predict_proba(Xte)[:, 1]
                                                 if cong is not None else
                                                 (pred_util > C.SURGE_UTIL).astype(float)))
        self.test_df = test
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES]
        out = df.copy()
        out["pred_sessions"] = np.clip(self.models["load"].predict(X), 0, None)
        out["pred_utilization"] = np.clip(self.models["util"].predict(X), 0, 1)
        cong = self.models["congestion"]
        out["pred_congestion_prob"] = (cong.predict_proba(X)[:, 1] if cong is not None
                                       else (out["pred_utilization"] > C.SURGE_UTIL).astype(float))
        return out

    def feature_importance(self) -> pd.DataFrame:
        imp = self.models["load"].feature_importances_
        return (pd.DataFrame({"feature": FEATURES, "importance": imp})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
