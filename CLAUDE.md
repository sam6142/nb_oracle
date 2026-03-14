# The Neighborhood Oracle

Hyperlocal demand forecasting and store profitability advisor for small businesses.

## Product Philosophy

**Predictions are the foundation. Intelligence is the product.**

Don't compete with the owner's gut on obvious patterns (weekends are busy, storms need bread). Provide value through things he CAN'T know: cultural calendars, EBT/SNAP cycles, demographic-specific demand, sales diagnostics with dollar amounts.

Every output must have: a specific dollar amount, a specific action, and specific timing.

Target user: Tier 3 store owners with modern POS (Square/Clover/Toast), 6+ months of data, data-curious but no analytics skills.

## Repository Structure

```
config.py                    → Central config, all paths auto-detected from ROOT_DIR
setup_path.py                → Import in notebooks to fix module paths
dashboard.py                 → Streamlit dashboard (3 pages: Live Forecast, Store Dashboard, Alert Preview)
generate_boston_data.py       → Generates mock Boston bodega data with real weather

feature_store/
  engineer.py                → build_features(), build_prediction_features(), validate_features()
  sources/
    weather.py               → Open-Meteo API (historical + 7-day forecast, free, no key)
    events.py                → Boston events with 3-tier geographic impact model

model/
  train.py                   → train_and_save(), load_model(), train_all_categories()
  evaluate.py                → compute_wmape()
  registry/                  → Saved .joblib models + metadata JSON + optuna params

explainability/
  translator.py              → SHAP → natural language with confidence levels

alerts/
  alert_engine.py            → Threshold logic (stockout, spike, dead stock)
  channels/                  → WhatsApp/SMS via Twilio (not yet wired up)

db/
  schema.sql                 → Full Postgres/PostGIS schema (not yet deployed)

data/raw/
  boston-bodega/              → Mock bodega data with real Boston weather
  store-sales-time-series-forecasting/  → Kaggle Favorita dataset
```

## Key Design Decisions

- **Feature store as a table**: `feature_vectors` table in schema.sql serves both training and inference
- **Tweedie regression**: `objective="reg:tweedie"` for zero-inflated sales data
- **Global model with store features**: Not per-store models (not enough data per store)
- **Temporal splits only**: NEVER random train/test splits. Always split by date.
- **No future leakage**: All lag features use shift() with positive values only
- **Predictions must be non-negative**: Always `np.maximum(pred, 0)`
- **One code path for features**: `build_prediction_features()` for inference uses the same `build_features()` as training
- **Events have 3 tiers**: nearby (<2mi) BOOST traffic, city (2-10mi) may REDUCE it, universal (holidays) affect all stores

## Active Dataset

Controlled by `ACTIVE_DATASET` in config.py. Options: "boston" (default) or "favorita".
Boston data path: `data/raw/boston-bodega/`
Favorita data path: `data/raw/store-sales-time-series-forecasting/`

## Model Performance

| Dataset | Model | WMAPE | Notes |
|---|---|---|---|
| Favorita (Store 44, Beverages) | XGBoost default | 13.0% | Real supermarket data |
| Favorita (All categories avg) | XGBoost default | 34.0% | 88% beat baseline |
| Favorita (Multi-store avg) | XGBoost default | 11.6% | 70% beat baseline |
| Boston (Beverages) | XGBoost Optuna-tuned | 32.1% | Mock data with real weather |
| Boston (5 categories avg) | XGBoost Optuna-tuned | 26.4% | |

## Known Issues (see ISSUES.md for full list)

### 🔴 Still Open
- #4: No data validation (must implement 6 dimensions: accuracy, completeness, consistency, timeliness, uniqueness, validity)

### ✅ Fixed
- #1: Hardcoded paths → now auto-detected in config.py
- #2: No train/save/load → model/train.py with train_and_save() and load_model()
- #3: Training/serving skew → build_prediction_features() uses same pipeline as training

### Bugs
- Dashboard shows "excellent prediction" when actual is 0 (delta_pct defaults to 0)
- Confidence says "high" on bad predictions (based on feature agreement, not real uncertainty)

## What To Build Next (priority order)

1. **Data quality validation** (Issue #4) — 6 dimensions, every data source checked before pipeline
2. **Sales drop diagnostic** — "Why was last week bad?" with $ breakdown by factor using SHAP
3. **Daily summary message** — One message covering all categories with actions (replaces per-category alerts). Includes EBT/SNAP timing, weather context, cross-category recommendations when relevant.
4. **Dead stock detection** — flag items with near-zero velocity, estimate $ trapped
5. **Stockout detection** — pattern-based detection from daily sales anomalies
6. **Waste estimation** — temperature + shelf life profiles → estimated $ waste
7. **Cross-category recommendations** — hardcoded known product pairings (diapers → baby food, charcoal → plates, etc.)
8. **Storm/extreme weather playbook** — before/during/after timeline with category-specific actions

## Conventions

- All config in `config.py` — no hardcoded paths or magic numbers elsewhere
- Use `config.get_model_params(category)` to load model params (checks for per-category Optuna params first)
- Use `config.get_dataset_path()` for data paths
- Notebooks: `import setup_path` as first line, then import from project modules
- Windows: use `copy` not `cp`, `python` not `python3`, activate venv with `venv\Scripts\activate`
- Always update `UPDATES.md` after meaningful project changes
- Always update `ISSUES.md` when fixing or discovering issues
- Replace deprecated `pandas .last()` with index filtering: `df[df.index >= df.index.max() - pd.Timedelta(days=X)]`

## Running the Project

```bash
# Activate environment
venv\Scripts\activate

# Train all models
python -m model.train

# Train one category
python -m model.train --category BEVERAGES

# Run dashboard
streamlit run dashboard.py

# Generate Boston mock data
python generate_boston_data.py
```