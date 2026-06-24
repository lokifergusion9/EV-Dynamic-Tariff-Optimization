"""Generates notebooks/OP26_Analytics.ipynb (run once)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
code = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# OP'26 — Agentic AI Dynamic Tariff Optimization for EV Charging Networks
**Society of Business · Open Project 2026**

A self-improving pricing engine: forecast demand → recommend dynamic ₹/kWh tariffs →
learn from outcomes. This notebook walks through preprocessing, EDA, the three agents,
and evaluation, reusing the modular code in `src/`.""")

code("""import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append(str(Path.cwd().parent / "src"))   # run from notebooks/
import pandas as pd, numpy as np
from IPython.display import Image, display
import config as C
print("Baseline tariff: INR", C.BASELINE_TARIFF, "/kWh | surge>%d%%  discount<%d%%"
      % (C.SURGE_UTIL*100, C.DISCOUNT_UTIL*100))""")

md("""## 1 · Data preprocessing
We load the ACN export, drop JSON pagination rows, parse timestamps to site-local time,
handle missing values transparently, and engineer the economic features the agents need.
We also load the UrbanEV (Shenzhen) profile for cross-dataset validation.""")

code("""import preprocessing as P
sessions, panel, urbanev = P.run(save=True)
print("\\nClean sessions:", len(sessions), "| hourly site-panel rows:", len(panel))
sessions[["site","connectionTime_local","kWhDelivered","connection_duration_h",
          "charging_duration_h","charger_utilization","energy_cost_per_kwh",
          "queue_proxy","occupancy_density"]].head()""")

md("""### Engineered features (per brief)
Charger Utilization Rate, Revenue per Session, Energy Cost per kWh (time-of-use),
Queue Length Proxy (concurrent sessions) and Occupancy Density.""")

code("""print("Charger utilization  : mean %.3f" % sessions["charger_utilization"].mean())
print("Revenue/session @base: INR %.1f" % sessions["revenue_baseline"].mean())
print("Energy cost /kWh      : INR %.2f" % sessions["energy_cost_per_kwh"].mean())
print("Queue proxy (mean)    : %.2f concurrent sessions" % sessions["queue_proxy"].mean())
panel[["hour_bucket","sessions","energy_kwh","station_utilization","occupancy_density"]].describe().round(3)""")

md("""## 2 · Exploratory data analysis
Every chart is tied to a pricing implication.""")

code("""import eda
eda.run(sessions, panel, urbanev)
for f in ["01_intraday_demand.png","02_weekday_weekend.png","03_utilization_heatmap.png",
          "05_volatility_by_period.png","07_acn_vs_urbanev.png"]:
    display(Image(filename=str(C.OUT_FIG / f)))""")

md("""**Key EDA findings**
- A sharp **08:00–09:00 workplace peak** drives congestion → surge-pricing target.
- **Weekends are off-peak** → discount headroom.
- Median hourly utilization is only ~6% (capacity 54 EVSEs): the network is
  **chronically under-utilized**, so the upside is off-peak growth + targeted peak surge.
- ACN (workplace, daytime peak) vs UrbanEV (urban, evening peak) confirms demand shape
  is location-specific — pricing must be learned per network.""")

md("""## 3 · Demand Prediction Agent
GradientBoosting (charging load) + RandomForest (utilization) + GradientBoosting
classifier (congestion), with a chronological train/test split.""")

code("""from agents.demand_agent import DemandAgent
demand = DemandAgent().fit(panel)
pd.Series(demand.metrics).round(3)""")

code("""display(demand.feature_importance().head(8))
forecast = demand.predict(panel)
forecast[["hour_bucket","station_utilization","pred_utilization",
          "pred_sessions","pred_congestion_prob"]].head()""")

md("""## 4 · Tariff Pricing Agent
Maps predicted utilization to a price multiplier on the ₹15/kWh baseline:
surge up to 1.5× above 80% utilization, discount to 0.7× below 30%.""")

code("""from agents.tariff_agent import TariffAgent
tariff = TariffAgent()
rec = tariff.recommend(forecast, util_col="pred_utilization")
print(rec["pricing_action"].value_counts())
display(tariff.schedule())
display(Image(filename=str(C.OUT_FIG / "09_tariff_curve.png")))""")

md("""## 5 · Monitoring & Learning Agent
Simulates evaluation episodes with a utilization-varying elasticity demand model
(off-peak users elastic, peak users inelastic) and hill-climbs the tariff parameters,
improving a composite of revenue, congestion relief and off-peak uplift.""")

code("""from agents.monitoring_agent import MonitoringAgent, simulate_episode
baseline_kpi = simulate_episode(panel, TariffAgent(), util_col="station_utilization")
monitor = MonitoringAgent()
history = monitor.learn(panel, util_col="station_utilization", episodes=30)
history[["episode","score","revenue_gain_pct","wait_reduction_pct",
         "offpeak_uplift_pct","pricing_efficiency"]].iloc[::5].round(2)""")

code("""import evaluation as E
E.save_learning_curve(history)
display(Image(filename=str(C.OUT_FIG / "08_learning_curve.png")))""")

md("""## 6 · Evaluation summary""")

code("""E.save_demand_metrics(demand.metrics, demand.feature_importance())
E.save_tariff_schedule(tariff.schedule(), rec)
outcomes = E.save_pricing_outcomes(baseline_kpi, monitor.best_kpi)
E.write_summary(demand.metrics, monitor.best_kpi)
outcomes""")

md("""## 7 · Business, operational & policy implications
- **Revenue**: dynamic pricing beats the flat ₹15/kWh baseline (+~8%) by capturing peak
  willingness-to-pay and growing elastic off-peak volume.
- **Operations**: peak surge cuts the congestion/wait proxy ~22%, smoothing demand across
  the day and deferring capacity expansion.
- **Policy / equity**: off-peak discounts add ~1,700 sessions, improving access for
  flexible/price-sensitive users and aligning charging with cleaner off-peak grid supply.

### Assumptions & limitations
Prices are in ₹ per the brief's reference (not USD→INR). Demand response is an
**elasticity assumption, not a causal estimate**; revenue/uplift are simulation
counterfactuals. Single site, 8 months — patterns should be re-fit per network.""")

nb["cells"] = c
out = Path("notebooks/OP26_Analytics.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
