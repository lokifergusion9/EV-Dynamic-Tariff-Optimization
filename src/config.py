"""Central configuration for the OP'26 Agentic Tariff Optimization pipeline.

All paths, business constants, and modeling thresholds live here so every stage
(preprocessing -> EDA -> agents -> evaluation) reads from a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # op26_project/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_CSV = ROOT / "outputs" / "csv"

# Source datasets
ACN_XLSX = ROOT.parent / "acndata_sessions.json.xlsx"      # provided file (one level up)
URBANEV_DIR = DATA_RAW / "ST-EVCDP" / "datasets"           # cloned UrbanEV repo

for _d in (DATA_RAW, DATA_PROCESSED, OUT_FIG, OUT_CSV):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Business / pricing constants  (per OP'26 brief)
# --------------------------------------------------------------------------- #
BASELINE_TARIFF = 15.0        # INR per kWh  -- fixed-rate baseline to beat
CURRENCY = "INR"

# Dynamic pricing band (multipliers on the baseline tariff)
SURGE_UTIL = 0.80             # utilization above this -> surge pricing
DISCOUNT_UTIL = 0.30          # utilization below this -> discount pricing
MAX_MULTIPLIER = 1.50         # cap on surge
MIN_MULTIPLIER = 0.70         # floor on discount

# Time-of-use energy procurement cost proxy (INR/kWh) -- what the operator pays
# the grid. Higher during the evening grid peak, cheaper overnight. Documented
# assumption used for margin / pricing-efficiency analysis.
TOU_PEAK_HOURS = set(range(17, 22))        # 17:00-21:59 grid peak
TOU_OFFPEAK_HOURS = set(range(0, 6))       # 00:00-05:59 grid off-peak
ENERGY_COST_PEAK = 9.0
ENERGY_COST_SHOULDER = 7.0
ENERGY_COST_OFFPEAK = 5.0

# Demand-period definitions (local hour of day) -- used across EDA & agents
PEAK_HOURS = set(range(8, 18))             # workplace charging peak (08:00-17:59)
OFFPEAK_HOURS = set(list(range(0, 7)) + list(range(21, 24)))

# Price-elasticity assumption (documented, NOT a causal claim).
# demand_factor = 1 + elasticity * (price_multiplier - 1).
# EV pricing reality: peak/workday users are time-constrained -> inelastic;
# off-peak users are flexible -> elastic (|e|>1), so off-peak discounts grow
# volume more than they cut price (revenue-positive off-peak uplift).
PEAK_ELASTICITY = -0.20        # high-utilization hours (inelastic)
OFFPEAK_ELASTICITY = -1.45     # low-utilization hours (elastic)
PRICE_ELASTICITY = -0.35       # blended default (back-compat)

# Congestion comfort threshold: utilization above this creates queueing/wait.
CONGESTION_UTIL = 0.60

# Reproducibility
RANDOM_STATE = 42

# Capacity assumptions for utilization/occupancy when station counts are sparse
MIN_STATIONS_PER_SITE = 1
