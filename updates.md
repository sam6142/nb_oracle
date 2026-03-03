# Updates — The Neighborhood Oracle

A running log of everything we learned while building this project.
Updated every time we hit a problem, make a correction, or discover something useful.

---

## Session 1 — March 2-3, 2026

### Setup & Environment

**1. Git isn't always available in the terminal on Windows.**
GitHub Desktop app does NOT install the `git` command for your terminal. You need to install Git separately from [git-scm.com](https://git-scm.com). During installation, choose "Git from the command line and also from 3rd-party software" so VS Code's terminal can find it.

**2. Windows PowerShell blocks Python virtual environment activation by default.**
Running `venv\Scripts\activate` gives a "running scripts is disabled" error. Fix it once with:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. Always activate your venv before doing anything.**
If you don't see `(venv)` at the start of your terminal line, your packages won't be found. Every time you reopen VS Code, run `venv\Scripts\activate` again.

**4. VS Code can run Jupyter notebooks natively.**
You don't need to open Jupyter in a browser. Just create a `.ipynb` file in VS Code and it works. Make sure to select the venv kernel (top right of notebook), not the system Python.

**5. When VS Code asks for a Jupyter server password, you did something wrong.**
Don't paste notebook URLs into VS Code. Just create `.ipynb` files directly in the sidebar. VS Code handles the Jupyter server automatically.

**6. Select the venv kernel, not system Python.**
When VS Code shows kernel options, pick the one that mentions `venv`. If it doesn't appear, use Ctrl+Shift+P → "Python: Select Interpreter" → "Enter interpreter path" → navigate to `venv\Scripts\python.exe`.

---

### Git & GitHub

**7. `git commit -m "message"` — the message is a label for your snapshot.**
It describes WHAT you changed. Keep it short, start with a verb. Examples: "Add XGBoost model", "Fix feature engineering bug", "Update README".

**8. `.gitignore` must be created BEFORE your first commit.**
Otherwise Git starts tracking files you don't want (like `.env` secrets or large data files). We caught this in time.

**9. Data files don't go on GitHub.**
Our Favorita dataset is 100MB+. The `.gitignore` blocks `data/raw/` and `data/processed/` for this reason. Anyone cloning the repo downloads the data separately from Kaggle.

---

### Data & Feature Engineering

**10. `pandas .last()` was removed in newer versions.**
If you get `AttributeError: 'Series' object has no attribute 'last'`, replace:
```python
# Old (broken):
bev.last("90D")

# New (works):
bev[bev.index >= bev.index.max() - pd.Timedelta(days=90)]
```

**11. Always build a dumb baseline before building anything fancy.**
Our baseline (average of last 4 same-weekday sales) got 12.3% average WMAPE. This is the number every model must beat. If XGBoost can't beat a simple average, the features need work, not the model.

**12. Never use random train/test splits on time series data.**
Always split by date. Training data = everything before date X. Test data = everything after date X. Random splits let the model "cheat" by learning from future data.

**13. Sales data is zero-inflated — use Tweedie regression.**
Many product categories have 80-90% zero-sale days. Regular regression (MSE) would just predict "0" for everything. Tweedie regression handles this pattern properly. Set `objective="reg:tweedie"` in XGBoost.

**14. Lag features must only look backward — never forward.**
Every `shift()` call must be positive (looking at past data). Using `shift(-1)` would peek at tomorrow's sales, which is cheating. This is called "data leakage" and it makes your metrics look great but your model useless in production.

**15. It's not the absolute temperature that matters — it's the change.**
A 90°F day after a week of 90°F is normal. A 90°F day after a week of 70°F drives different buying behavior. Always create delta features: `temp_delta_vs_yesterday` and `temp_delta_vs_weekly_avg`.

---

### Model Building

**16. XGBoost's most important feature was `is_weekend` (0.52 importance).**
More than half the model's decision-making came from "is it a weekend?" The weekly cycle dominates small-store sales data. Day of week + recent sales history accounts for the vast majority of predictive power.

**17. Simulated weather gives small improvements — that's expected.**
With fake weather data, the model improved only 1.3% (13.2% → 13.0%). The weather features had near-zero SHAP values because they don't actually correlate with the real sales data. Real weather data for a real store would show much bigger improvement.

**18. WMAPE is better than MAPE for sales data.**
MAPE (Mean Absolute Percentage Error) breaks when actual sales = 0 (division by zero). WMAPE (Weighted MAPE) divides total error by total sales, avoiding this problem. Formula: `sum(|actual - predicted|) / sum(|actual|)`.

---

### Code Organization

**19. Don't copy-paste code between notebooks — use shared modules.**
We moved `build_features()` into `feature_store/engineer.py` and `compute_wmape()` into `model/evaluate.py`. Now every notebook imports them with one line instead of duplicating 50 lines of code. If you fix a bug, it's fixed everywhere.

**20. `sys.path.insert(0, str(Path.cwd().parent))` is needed in notebooks.**
Notebooks run from their own directory. To import from project modules (like `feature_store.engineer`), you need to add the project root to Python's search path. This line does that.

**21. config.py is the single source of truth for all settings.**
Model parameters, file paths, API keys, alert thresholds — all in one file. No magic numbers scattered across modules. When you need to change something, you change it in one place.

---

### Dashboard & Product

**22. Streamlit asks for your email on first run — just press Enter to skip.**
It's for their newsletter. Not required.

**23. Don't hit "Deploy" on Streamlit yet.**
It tries to publish to the internet but won't work because data files are local and too large for free hosting. Deployment needs proper setup later.

**24. `@st.cache_data` and `@st.cache_resource` prevent re-running expensive code.**
Without caching, Streamlit would reload 3M rows and retrain the model every time you move the date slider. Caching stores the result so it only runs once.

**25. The "Why" explanation is the product, not the prediction number.**
Store owners don't care that the model predicts 14,576 units. They care that "it's a Sunday, sales have been trending up, and it's payday." The SHAP → plain English translation is what makes this useful.

---

### Project Management

**26. Keep a PRODUCT_SPEC.md as the single source of truth.**
Lists all input features (built, planned, future), all output features, benchmarks, tech stack, and architecture decisions. Update it as you build. Any conversation can reference it.

**27. CLAUDE.md gives AI coding assistants project context.**
Put your architecture rules, conventions, and tech decisions in this file. Claude Code reads it automatically when working on your project.

---

**28. XGBoost beats baseline in 88% of categories at Store 44.**
Best: Eggs (6.0%), Dairy (6.4%), Bread/Bakery (6.5%). Worst: Magazines (20.8%), Electronics (19.0%). Categories with high zero-sale percentages (37%+) are hardest to predict.

**29. Cross-store average WMAPE is 11.6% across 10 stores and 5 categories.**
XGBoost beats baseline in 70% of store-category combos. Some stores (30, 54) are outliers — likely need location-specific features to improve.

**30. Dairy is the most predictable category across all stores. Beverages is the most variable.**
Beverages are more sensitive to external factors (weather, events, holidays), which is exactly why contextual features matter.

**31. Notebooks run from the `notebooks` directory even when the file is in root.**
Use `sys.path.insert(0, str(Path.cwd().parent))` not `sys.path.insert(0, str(Path.cwd()))` to import project modules.

**32. Dashboard v2 has three pages: Store Dashboard, Multi-Store Overview, and Alert Preview.**
The Multi-Store page trains models for all selected store-category combos and shows a color-coded heatmap, performance charts, and automated insights. The Alert Preview shows what WhatsApp messages would look like.

**33. When testing the Multi-Store page, select multiple stores and categories.**
Selecting just one store and one sparse category (like Automotive) gives misleading results. Use the core categories (Beverages, Dairy, Grocery I, Meats, Produce) across 5-10 stores for meaningful comparison.

**34. SHAP explanations upgraded from fact-listing to natural storytelling.**
Old: "Because your recent 7-day average (10,292) is above normal, it's a Sunday."
New: "Sundays are one of your busiest days — expect higher than average traffic. Your sales have been running strong this week." The translator module lives in `explainability/translator.py` and handles day-of-week, momentum, special events, and weather narratives.

**35. Confidence levels are derived from SHAP feature agreement.**
If 80%+ of top features push the same direction = HIGH. 60-80% = MODERATE. Below 60% = LOW. This gives store owners a trust signal.

**36. WhatsApp message formatter includes inventory shortfall warnings.**
`format_whatsapp_message()` takes a prediction + current inventory and generates actionable alerts like "Consider ordering 1,414 more."

**37. The full prediction-to-message pipeline works end to end.**
Raw data → feature engineering → XGBoost prediction → SHAP values → natural language translator → WhatsApp-formatted message with inventory shortfall check. Each step is a separate module that can be tested independently.


**38. Live weather forecasts work with Open-Meteo — no API key needed.**
fetch_weather_forecast() pulls the real 7-day forecast for any location. fetch_historical_weather() pulls past weather back to 1940. Both free.

**39. Events have three geographic tiers: nearby (< 2mi), city (2-10mi), universal.**
Nearby events BOOST store traffic. City events may REDUCE it by drawing customers elsewhere. A Red Sox game at Fenway (3.2mi) is different from a block party down the street (0.3mi). The event_score feature captures this net effect.

**40. Weather data length must match sales data length in build_features().**
When building features for a small window (like 60 days of recent history + 1 forecast day), filter the weather to only those dates. Don't pass the entire 793-row weather history or you get a length mismatch error.

**41. The full real-time pipeline works: historical data → train model → live weather + events → 7-day forecast with explanations.**
This is the production flow. Any store's POS data + retroactive weather + upcoming events = actionable predictions.

**42. Event features show strong signal but slightly hurt WMAPE (33.7% → 35.9%).**
nearby_attendance is the #2 most important feature (0.085), has_sports_nearby is #4 (0.062), event_score is #7 (0.040). The model recognizes events matter but with only 760 training days and 9 new features, it overfits. More training data (2+ years) would fix this.

**43. Adding too many features with limited data causes overfitting.**
Going from 20 to 35 features with ~760 rows is risky. The model has more knobs to turn than examples to learn from. In production with 2+ years of store data, event features would reliably improve predictions.

**44. Feature importance proves the concept even when WMAPE doesn't improve.**
If event features had zero importance, they'd be useless. But nearby_attendance at #2 means the model sees real signal. It just needs more data to use it reliably. This is a common pattern in ML — good features need sufficient training data to shine.

**45. Optuna hyperparameter tuning improved Beverages WMAPE by 12.4% (36.7% → 32.1%).**
100 trials, 3-5 minutes to run. Across 5 categories, average improvement was 2.8%. Three categories improved, two slightly worsened (overfitting to validation with limited data). Best params saved to model/registry/optuna_best_params.json.

**46. Optuna doesn't always help every category.**
Produce and Meats got slightly worse with tuned params. With only ~700 training days and 50 trials, Optuna can overfit to the validation window. More data and more trials would fix this. For production, tune per-category with at least 100 trials each.

**47. Three-way split is important for honest evaluation.**
Train / Validation / Test. Optuna sees validation but never test. Final numbers come from test only. If you tune on the same data you evaluate on, your numbers lie.
---

*This file will be updated as the project continues.*