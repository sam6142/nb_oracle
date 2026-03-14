"""
Central configuration for The Neighborhood Oracle.

All paths, settings, and parameters live here.
No hardcoded absolute paths anywhere else in the project.

ROOT_DIR is detected automatically — works on any machine.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths (auto-detected) ───────────────────────────────────────
# ROOT_DIR = the folder where this config.py file lives
# Everything else is relative to ROOT_DIR
ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
MODEL_REGISTRY = ROOT_DIR / "model" / "registry"

# ── Dataset paths ────────────────────────────────────────────────
# Switch between datasets by changing ACTIVE_DATASET
# Options: "favorita", "boston"
ACTIVE_DATASET = os.getenv("ACTIVE_DATASET", "boston")

DATASET_PATHS = {
    "favorita": DATA_RAW / "store-sales-time-series-forecasting",
    "boston": DATA_RAW / "boston-bodega",
}

def get_dataset_path():
    """Returns the path to the active dataset folder."""
    return DATASET_PATHS.get(ACTIVE_DATASET, DATASET_PATHS["boston"])

# ── Database Connection ──────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

# ── API Keys ─────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")

# ── Location (for weather and events) ────────────────────────────
STORE_LAT = float(os.getenv("STORE_LAT", "42.3601"))   # Boston default
STORE_LON = float(os.getenv("STORE_LON", "-71.0589"))
STORE_TIMEZONE = os.getenv("STORE_TIMEZONE", "America/New_York")

# ── Feature Engineering Settings ─────────────────────────────────
FORECAST_HORIZONS = [1, 3, 7]
LAG_WINDOWS = [7, 14, 28, 56]
PAYDAYS = [1, 15]

# ── Model Settings ───────────────────────────────────────────────
XGBOOST_DEFAULT_PARAMS = {
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

# ── Data Validation Thresholds ───────────────────────────────────
# Sales: above this many units/day is suspicious for a single bodega category
VALIDATION_MAX_DAILY_SALES = int(os.getenv("VALIDATION_MAX_DAILY_SALES", 10_000))
# Temperature bounds in Fahrenheit
VALIDATION_MAX_TEMP_F = 150
VALIDATION_MIN_TEMP_F = -40
# Daily precipitation above this is extreme but physically possible
VALIDATION_MAX_PRECIP_MM = 500
# Warn if the most recent row in sales/weather data is older than this
VALIDATION_STALE_SALES_DAYS = int(os.getenv("VALIDATION_STALE_SALES_DAYS", 2))
VALIDATION_STALE_WEATHER_DAYS = int(os.getenv("VALIDATION_STALE_WEATHER_DAYS", 2))

# ── Alert Settings ───────────────────────────────────────────────
ALERT_SPIKE_FACTOR = float(os.getenv("ALERT_DEMAND_SPIKE_FACTOR", 1.3))
ALERT_DEAD_STOCK_FACTOR = float(os.getenv("ALERT_DEAD_STOCK_FACTOR", 0.5))
ALERT_MAX_PER_DAY = 5


def get_model_params(category=None):
    """
    Load the best model params for a category.
    Checks for Optuna-tuned params first, falls back to defaults.
    """
    import json
    
    if category:
        # Try category-specific params first
        cat_path = MODEL_REGISTRY / f"optuna_params_{category.lower().replace(' ', '_').replace(',', '')}.json"
        if cat_path.exists():
            with open(cat_path) as f:
                return json.load(f)
    
    # Try general Optuna params
    general_path = MODEL_REGISTRY / "optuna_best_params.json"
    if general_path.exists():
        with open(general_path) as f:
            return json.load(f)
    
    # Fall back to defaults
    return XGBOOST_DEFAULT_PARAMS.copy()