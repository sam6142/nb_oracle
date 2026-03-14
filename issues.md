# Known Issues & Technical Debt — The Neighborhood Oracle

Ranked by priority. Fix the top ones first. Ignore the bottom ones for now.

---

## 🔴 FIX NOW (blocks credibility and demo quality)

### 1. Hardcoded absolute paths everywhere
**Impact:** Nobody else can run your code. Deployment impossible.
**Where:** dashboard.py, every notebook, generate_boston_data.py
**Fix:** Move all paths into config.py using relative paths or environment variables. One change in config.py should make everything work on any machine.
**Time:** 1-2 hours

### 2. No train/save/load pipeline — model retrains every dashboard load
**Impact:** Slow dashboard startup. No model versioning. No rollback if a retrain goes bad.
**Where:** dashboard.py retrains on every page load (cached, but still wrong architecture)
**Fix:** Build model/train.py that trains once, saves to model/registry/ as .joblib files. Dashboard loads the saved model instead of training. Separate training from serving.
**Time:** 2-3 hours

### 3. Training/serving skew in feature engineering
**Impact:** Live predictions may use slightly different features than training, causing silent errors.
**Where:** make_live_prediction() in dashboard.py builds features differently than the training path. Missing features are silently filled with 0.0.
**Fix:** One single function should produce features for both training and inference. Add schema validation — if the feature list doesn't match exactly, raise an error.
**Time:** 2-3 hours

### ✅ 4. No data validation — FIXED
**Fix applied:** `data/validation.py` — 6-dimension checks (accuracy, completeness, consistency,
timeliness, uniqueness, validity) for sales, weather, and events data.
Integrated into `model/train.py`: errors abort training; warnings print but allow training to continue.
Weather/events validation failures drop those features instead of aborting entirely.

---

## 🟡 FIX SOON (important for project quality, not urgent)

### 5. Notebooks are messy and not reproducible
**Impact:** If someone clones the repo and runs notebooks in order, they'll hit errors. Files are in root instead of notebooks/ folder. Data paths vary between notebooks.
**Where:** 01_EDA.ipynb through 12_optuna_tuning.ipynb
**Fix:** Move all notebooks to notebooks/ folder. Standardize the import/path setup. Add a README in notebooks/ explaining run order and dependencies. Use setup_path.py consistently.
**Time:** 1-2 hours

### 6. No tests
**Impact:** Changes to build_features() or the model can silently break things. No safety net.
**Where:** tests/ folder is empty.
**Fix:** Add 3-5 critical tests: (a) lag features don't leak future data, (b) feature schema matches model expectations, (c) WMAPE calculation is correct, (d) predictions are non-negative, (e) SHAP translator produces valid output.
**Time:** 2-3 hours

### 7. Optuna params saved globally but should be per-category
**Impact:** One set of tuned params applied to all 33 categories. Hurts Produce and Meats (proven in testing).
**Where:** model/registry/optuna_best_params.json is one file for all categories.
**Fix:** Save params per category: optuna_params_beverages.json, optuna_params_dairy.json, etc. Dashboard loads the right one based on category selection. Fall back to defaults for un-tuned categories.
**Time:** 1-2 hours

### 8. No proper cross-validation
**Impact:** Model evaluation is based on one 30-day window. Could be misleadingly good or bad.
**Where:** Every notebook uses a single temporal split.
**Fix:** Implement expanding window CV: train on months 1-6 test on 7, train on 1-7 test on 8, etc. Report mean and std of WMAPE across windows.
**Time:** 2-3 hours

### 9. No prediction intervals
**Impact:** Store owner sees "106 units" with no sense of how uncertain that is. Confidence score (high/moderate/low) is based on SHAP agreement, not actual prediction uncertainty.
**Where:** explainability/translator.py confidence assessment.
**Fix:** Add quantile regression (train 10th and 90th percentile models) or conformal prediction. Report "90-120 units" instead of just "106 units."
**Time:** 3-4 hours

### 10. Duplicate model training code across 8+ files
**Impact:** If you change training logic, you have to change it in 8 places. Easy to miss one.
**Where:** Notebooks 03, 04, 05, 06, 07, 10, 11, 12 + dashboard.py all have XGBRegressor training code.
**Fix:** Create a train_and_evaluate() function in model/train.py. All notebooks and dashboard call this one function.
**Time:** 2-3 hours

---

## 🟢 CAN WAIT (nice to have, not blocking anything)

### 11. Boston sales data has built-in leakage
**Impact:** Weather-sales correlation is artificially strong because we generated sales using weather. Inflates feature importance for weather features. Favorita results are more honest.
**Where:** generate_boston_data.py
**Fix:** Be transparent about this in any demo. For production, this goes away automatically because you'll use real store data. Could also add noise to reduce the artificial correlation.
**Time:** Not critical to fix

### 12. No feedback loop from store owners
**Impact:** The model never learns from its own predictions. No way to track if owners acted on alerts.
**Where:** Not built at all.
**Fix:** Add a simple "was this helpful?" button in WhatsApp alerts, or track prediction vs actual over time per store. This is a V2 feature.
**Time:** Large effort, design needed

### 13. No logging or prediction audit trail
**Impact:** When something goes wrong, no record of what happened. Can't debug production issues.
**Where:** Nowhere — no logging exists.
**Fix:** Log every prediction: timestamp, input features, output, model version, confidence. Use Python's logging module. Store in a predictions table (schema already exists in schema.sql).
**Time:** 2-3 hours

### 14. Alert throttling not connected
**Impact:** The dashboard has alert settings sliders but they don't actually do anything.
**Where:** dashboard.py Alert Preview page.
**Fix:** Wire up the sliders to filter which alerts get shown. In production, the alert_engine.py already has throttling logic — just needs to be connected.
**Time:** 1 hour

### 15. No cold-start solution for new stores
**Impact:** New stores have no history. Model needs 28+ days of data. First month = no predictions.
**Where:** Not addressed anywhere.
**Fix:** Options include: neighborhood-based priors, transfer learning from similar stores, or simple category averages until enough data accumulates. Design decision needed.
**Time:** Large effort, research needed

### 16. WMAPE is misleading for low-volume categories
**Impact:** Magazines (2 units/day) shows 20% WMAPE but the error is only 0.4 units. Sounds terrible but is meaningless.
**Where:** All evaluation.
**Fix:** Add secondary metrics: MAE (mean absolute error in units) alongside WMAPE. For low-volume categories, report direction accuracy instead of percentage error.
**Time:** 1-2 hours

### 17. No deployment path
**Impact:** Dashboard only runs locally.
**Where:** dashboard.py reads from local file paths.
**Fix:** Containerize with Docker, use relative paths, deploy to Streamlit Cloud or Railway. Requires fixing issue #1 (hardcoded paths) first.
**Time:** 3-4 hours (after #1 is fixed)

### 18. Autonomous procurement has no foundation
**Impact:** Product spec promises it but zero infrastructure exists.
**Where:** Product spec only.
**Fix:** This is a separate product, not a feature. Requires distributor integration, pack sizes, MOQs, lead times. Don't promise it on a near-term roadmap. Focus on predictions and alerts first.
**Time:** Months of work — defer entirely

---

## Summary

| Priority | Count | Action |
|---|---|---|
| 🔴 Fix Now | 4 | Do these before any new features |
| 🟡 Fix Soon | 6 | Do these before going to production |
| 🟢 Can Wait | 8 | Nice to have, won't block you |

**Rule of thumb:** If you're tempted to add a new feature, check this list first. Fixing a 🔴 issue is always more valuable than adding a new bell or whistle.

---

*Last reviewed: March 4, 2026*