# The Neighborhood Oracle

Hyperlocal demand forecasting and store profitability advisor for small businesses.

---

## Auto-Update Rules

After completing any task that meaningfully changes the project, update these files as needed:

- `CLAUDE.md` — Architecture, conventions, datasets, or structure changed
- `UPDATES.md` — Always. Add a numbered entry with what was done and lessons learned
- `ISSUES.md` — Bug fixed (mark ✅) or new issue discovered (add it)
- `product_specs.md` — Input features, output features, or benchmarks changed
- `PRODUCT_DIRECTION.md` — Product strategy or build priorities changed
- `README.md` — Project status checklist items completed
- `requirements.txt` — New Python package added
- `.env.example` — New API key or environment variable added
- `config.py` — New settings, paths, or parameters added

Do NOT ask for permission to update these files. Just do it as part of completing the task.

## Commit Rules

- Do NOT auto-commit or auto-push to git. Only make changes to files.
- I will review changes and commit manually when I'm satisfied.
- If I say "commit this", then create a commit with a descriptive message.

## When You Hit an Error

- Show me the full error message and explain what caused it
- Propose a fix and explain WHY it will work
- Do NOT try multiple random fixes without explaining each one
- If unsure, ask me before making changes

## Code Style

- Add comments explaining WHY, not WHAT (the code shows what, comments explain why)
- Keep functions under 50 lines. If longer, break into smaller functions
- Every new function needs a docstring explaining what it does, parameters, and returns
- Use descriptive variable names (not x, df2, temp)
- Simple, step-by-step explanations without assuming prior technical knowledge — but never limit the project's scope or ambition

## Do NOT Modify

- Notebooks 01-07 in notebooks/ folder (old Favorita analysis, kept for reference)
- data/raw/ contents (generated or downloaded data)
- venv/ folder
- model/registry/*.joblib files (trained models)

## After Making Changes

- Run `streamlit run dashboard.py` to verify dashboard still works
- If you changed feature_store/engineer.py, run `python -m model.train --category BEVERAGES` to verify training still works
- If you changed model/train.py, verify load_model() still works
- If you added a new package, add it to requirements.txt

---

## Environment

- OS: Windows 11
- Python: 3.11 (use `python` not `python3`)
- Terminal: PowerShell in VS Code
- Venv activation: `venv\Scripts\activate`
- Set execution policy if needed: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Copy command: `copy` not `cp`
- Path separator: `\` not `/`
- Replace deprecated `pandas .last()` with index filtering: `df[df.index >= df.index.max() - pd.Timedelta(days=X)]`

---

## Product Philosophy

**Predictions are the foundation. Intelligence is the product.**

Don't compete with the owner's gut on obvious patterns (weekends are busy, storms need bread). Provide value through things he CAN'T know: cultural calendars, EBT/SNAP cycles, demographic-specific demand, sales diagnostics with dollar amounts.

Every output must have: a specific dollar amount, a specific action, and specific timing.

Target user: Tier 3 store owners with modern POS (Square/Clover/Toast), 6+ months of data, data-curious but no analytics skills. ~50-100K stores in the US.

See `PRODUCT_DIRECTION.md` for full product strategy.

---

## Repository Structure

```
config.py                    → Central config, all paths auto-detected from ROOT_DIR
setup_path.py                → Import in notebooks to fix module paths
dashboard.py                 → Streamlit dashboard (3 pages: Live Forecast, Store Dashboard, Alert Preview)
generate_boston_data.py       → Generates mock Boston bodega data with real weather

feature_store/
  engineer.py                → build_features(), build_prediction_features(), validate_features()
  sources/
    weather.py               → Open-Meteo API (historical + 7-day forecast, free, no key needed)
    events.py                → Boston events with 3-tier geographic impact model

data/
  validation.py              → validate_sales_data(), validate_weather_data(), validate_events_data()

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

notebooks/                   → Jupyter notebooks 01-12 (reference, do not modify 01-07)

data/raw/
  boston-bodega/              → Mock bodega data with real Boston weather + events
  store-sales-time-series-forecasting/  → Kaggle Favorita dataset (Ecuador)

Reference docs:
  CLAUDE.md                  → This file (project context for Claude Code)
  PRODUCT_DIRECTION.md       → Product strategy and build priorities
  product_specs.md           → Technical spec (input/output features, benchmarks)
  ISSUES.md                  → Known bugs and technical debt (prioritized)
  UPDATES.md                 → Running log of all changes and lessons learned
```

---

## Key Design Decisions

- **Feature store as a table**: `feature_vectors` table in schema.sql serves both training and inference
- **Tweedie regression**: `objective="reg:tweedie"` for zero-inflated sales data
- **Global model with store features**: Not per-store models (not enough data per store)
- **Temporal splits only**: NEVER random train/test splits. Always split by date
- **No future leakage**: All lag features use shift() with positive values only
- **Predictions must be non-negative**: Always `np.maximum(pred, 0)`
- **One code path for features**: `build_prediction_features()` for inference uses the same `build_features()` as training. Never build features manually.
- **Events have 3 tiers**: nearby (<2mi) BOOST traffic, city (2-10mi) may REDUCE it, universal (holidays) affect all stores
- **Train/serve separation**: model/train.py trains and saves models. Dashboard loads saved models. Never train in the dashboard.
- **Config is the single source of truth**: All paths, params, API keys live in config.py. No hardcoded values elsewhere.

---

## Active Dataset

Controlled by `ACTIVE_DATASET` in config.py (or `.env`). Options: "boston" (default) or "favorita".

- Boston data: `data/raw/boston-bodega/` — mock bodega sales + real Boston weather + simulated events
- Favorita data: `data/raw/store-sales-time-series-forecasting/` — real Ecuador supermarket data from Kaggle

Both datasets have identical CSV format (train.csv, stores.csv, holidays_events.csv) so all code works with either.

---

## Model Performance

| Dataset | Model | WMAPE | Notes |
|---|---|---|---|
| Favorita (Store 44, Beverages) | XGBoost default | 13.0% | Real supermarket data |
| Favorita (All categories avg) | XGBoost default | 34.0% | 88% beat baseline |
| Favorita (Multi-store, 10 stores) | XGBoost default | 11.6% | 70% beat baseline |
| Boston (Beverages) | XGBoost Optuna-tuned | 32.1% | Mock data, real weather |
| Boston (5 categories avg) | XGBoost Optuna-tuned | 26.4% | |

---

## Known Issues (see ISSUES.md for full prioritized list)

### 🔴 Still Open

### ✅ Fixed
- #1: Hardcoded paths → auto-detected in config.py
- #2: No train/save/load → model/train.py with train_and_save() and load_model()
- #3: Training/serving skew → build_prediction_features() uses same pipeline
- #4: No data validation → data/validation.py with 6-dimension checks (accuracy, completeness, consistency, timeliness, uniqueness, validity); integrated into model/train.py

### Known Bugs
- Dashboard shows "excellent prediction" when actual is 0 (delta_pct defaults to 0)
- Confidence says "high" on bad predictions (based on feature agreement, not real uncertainty — needs prediction intervals)
- Events don't improve WMAPE (35.9% vs 33.7% without) due to limited training data — but nearby_attendance is #2 feature by importance

---

## What To Build Next (priority order)

See `PRODUCT_DIRECTION.md` for full rationale.

1. **Data quality validation** (Issue #4) — 6 dimensions, every data source checked
2. **Sales drop diagnostic** — "Why was last week bad?" with $ breakdown by factor using SHAP. This is the #1 product feature.
3. **Daily summary message** — One message covering all categories with actions. Replaces per-category alerts. Includes EBT/SNAP timing, weather context, cross-category recommendations.
4. **Dead stock detection** — Flag items with near-zero velocity, estimate $ trapped on shelves
5. **Stockout detection** — Pattern-based detection from daily sales anomalies
6. **Waste estimation** — Temperature + shelf life profiles → estimated $ wasted
7. **Cross-category recommendations** — Hardcoded known product pairings (diapers → baby food, charcoal → plates, rice → cooking oil, etc.)
8. **Storm/extreme weather playbook** — Before/during/after timeline with category actions

---

## Running the Project

```bash
# Activate environment
venv\Scripts\activate

# Train all models (saves to model/registry/)
python -m model.train

# Train one category
python -m model.train --category BEVERAGES

# Run dashboard
streamlit run dashboard.py

# Generate Boston mock data (only needed once)
python generate_boston_data.py
```

---

## Key Functions Reference

```python
# Config
from config import get_dataset_path, get_model_params, MODEL_REGISTRY, STORE_LAT, STORE_LON

# Feature engineering
from feature_store.engineer import build_features, get_feature_columns, build_prediction_features, validate_features

# Weather (all free, no API key)
from feature_store.sources.weather import fetch_weather_forecast, fetch_historical_weather, generate_simulated_weather

# Events
from feature_store.sources.events import generate_boston_events, events_to_features

# Model
from model.train import train_and_save, load_model, train_all_categories, list_trained_models
from model.evaluate import compute_wmape

# Explainability
from explainability.translator import get_shap_explanation, format_whatsapp_message

# In notebooks, always start with:
import setup_path
```