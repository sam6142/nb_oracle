"""
Central configuration for The Neighborhood Oracle.

WHY THIS FILE EXISTS:
Instead of scattering numbers and settings across 20 different files,
we put them all here. If you ever want to change how the model works
or where data is stored, you only change THIS file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# This reads your .env file and makes the values available
load_dotenv()

# ── Paths (where things are stored on disk) ──────────────────────
ROOT_DIR = Path(__file__).parent          # The project root folder
DATA_RAW = ROOT_DIR / "data" / "raw"      # Raw downloaded data
DATA_PROCESSED = ROOT_DIR / "data" / "processed"  # Cleaned data
MODEL_REGISTRY = ROOT_DIR / "model" / "registry"  # Saved models

# ── Database Connection ──────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

# ── API Keys (also from .env) ────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ── Feature Engineering Settings ─────────────────────────────────
FORECAST_HORIZONS = [1, 3, 7]    # How many days ahead we predict
LAG_WINDOWS = [7, 14, 28, 56]   # How far back we look for patterns
PAYDAYS = [1, 15]                # Days of month when people get paid

# ── Model Settings ───────────────────────────────────────────────
XGBOOST_BASE_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.6,
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "early_stopping_rounds": 30,
}

OPTUNA_N_TRIALS = 150

VALIDATION_WMAPE_THRESHOLD = float(
    os.getenv("MODEL_RETRAIN_WMAPE_THRESHOLD", 0.15)
)

# ── Alert Settings ───────────────────────────────────────────────
ALERT_SPIKE_FACTOR = float(os.getenv("ALERT_DEMAND_SPIKE_FACTOR", 1.3))
ALERT_DEAD_STOCK_FACTOR = float(os.getenv("ALERT_DEAD_STOCK_FACTOR", 0.5))
ALERT_MAX_PER_DAY = 5