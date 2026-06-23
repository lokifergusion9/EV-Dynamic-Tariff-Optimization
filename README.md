# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

A self-improving pricing engine for EV charging networks. It forecasts demand,
recommends dynamic per-kWh tariffs, and learns from simulated outcomes to maximize
revenue while relieving peak congestion and lifting off-peak usage.

Built for **Open Project 2026 (Society of Business)**.

---

## 1. Data landscape

| Dataset | Role | Coverage | Notes |
|---|---|---|---|
| **ACN-Data** (`acndata_sessions.json.xlsx`) | Primary | 14,999 clean sessions, Caltech site (siteID 2), Apr–Dec 2018 | Real session timestamps, kWh delivered, station/space IDs |
| **UrbanEV / ST-EVCDP** (`data/raw/ST-EVCDP`) | Cross-validation | 8,640 × 247 zones, 5-min, Shenzhen, Jun 2022 | Occupancy/volume/price; used to compare intraday demand shapes |

The raw ACN export interleaves 1,305 JSON pagination/meta rows (no `connectionTime`)
which are dropped. `userInputs` is null for 100% of rows (dropped); `kWhRequested`
is ~78% missing (not used — `kWhDelivered` is the energy measure).

## 2. Pipeline

```
src/
  config.py            constants: ₹15/kWh baseline, surge>80%/discount<30%, elasticity, paths
  preprocessing.py     clean sessions, engineer features, hourly panel, UrbanEV profile
  eda.py               9 insight-driven figures -> outputs/figures/
  agents/
    demand_agent.py    ML forecast: charging load, utilization, congestion prob
    tariff_agent.py    utilization -> dynamic tariff (piecewise-linear multiplier)
    monitoring_agent.py feedback loop: simulate outcomes, hill-climb tariff params
  evaluation.py        metrics + summary CSVs/figures
  run_pipeline.py      end-to-end orchestrator
notebooks/OP26_Analytics.ipynb   narrative walk-through
outputs/figures/*.png            EDA + results charts
outputs/csv/*.csv                metrics, tariff schedule, learning episodes
```

### Run it
```bash
pip install -r requirements.txt
cd src
python run_pipeline.py            # preprocess -> EDA -> agents -> evaluation
# (optional) UrbanEV cross-dataset:
git clone --depth 1 https://github.com/IntelligentSystemsLab/ST-EVCDP.git data/raw/ST-EVCDP
```

## 3. Engineered features (per brief)
- **Charger Utilization Rate** = charging time / connection (occupied) time
- **Revenue per Session** = kWh delivered × tariff
- **Energy Cost per kWh** = time-of-use grid procurement proxy (peak/shoulder/off-peak)
- **Queue Length Proxy** = concurrent active sessions per site-hour (interval overlap)
- **Occupancy Density** = active sessions / distinct stations (capacity = 54 EVSEs)
- Plus calendar + lag/rolling features for the demand model.

## 4. The three agents
1. **Demand Prediction Agent** — GradientBoosting (load) + RandomForest (utilization) +
   GradientBoosting classifier (congestion), chronological train/test split.
2. **Tariff Pricing Agent** — maps predicted utilization to a price multiplier on the
   ₹15/kWh baseline: surge up to 1.5× above 80% utilization, discount to 0.7× below 30%.
3. **Monitoring & Learning Agent** — simulates evaluation episodes with a demand-response
   model and hill-climbs the tariff parameters, improving a composite of revenue,
   congestion relief and off-peak uplift over episodes.

## 5. Headline results (learned policy vs fixed ₹15/kWh)

| Metric | Value |
|---|---|
| Demand load model | R² 0.90, RMSE 0.95, MAE 0.61 |
| Utilization model | R² 0.80, RMSE 0.088 |
| Congestion classifier | Accuracy 0.98, ROC-AUC 0.96 |
| Revenue Gain % | **+8.2%** |
| Avg wait-time reduction (peak) | **−22%** |
| Off-peak uplift | **+1,700 sessions** |
| Pricing efficiency | **₹15.1 / kWh** |

Composite policy score improves from 4.8 → 23.6 over 30 episodes (the feedback loop
working). Exact numbers regenerate on each run; see `outputs/RESULTS_SUMMARY.txt`.

## 6. Assumptions & limitations (transparency)
- ACN sessions are US workplace charging; prices are expressed in ₹ using the brief's
  ₹15/kWh reference — not a USD→INR conversion.
- "Agentic" = autonomous ML-forecast → rule-based pricing → feedback-learning loop
  (deterministic, no external LLM/API).
- **Demand response is an elasticity assumption, not a causal estimate.** Off-peak
  users are modeled as price-elastic (|e|≈1.45), peak users inelastic (|e|≈0.20),
  reflecting standard EV-pricing flexibility patterns. Revenue/uplift figures are
  simulation-based counterfactuals.
- This export is a single site over 8 months; utilization is genuinely low
  (median 6%), so the optimization opportunity is off-peak growth + targeted peak surge.
