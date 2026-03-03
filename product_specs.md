
# The Neighborhood Oracle — Product Specification

## Last Updated: March 2, 2026

---

## INPUT FEATURES (What the model consumes)

### ✅ Built & Working
| Feature | Category | Data Source | Status |
|---|---|---|---|
| Historical Sales (lag features) | Internal Store Data | Kaggle (dev) / Supabase (prod) | ✅ Done |
| Product Category | Internal Store Data | Per-category model training | ✅ Done |
| Day of the Week | Contextual (Time) | System Clock | ✅ Done |
| Month / Season | Contextual (Time) | System Clock | ✅ Done |
| Holiday Status | Contextual (Time) | Calendar API / holidays CSV | ✅ Done |
| Promotions | Internal Store Data | onpromotion column | ✅ Done |

### 🟡 Next To Build
| Feature | Category | Data Source | Priority |
|---|---|---|---|
| Temperature | Contextual (Weather) | OpenWeatherMap API (free) | HIGH — next step |
| Precipitation | Contextual (Weather) | OpenWeatherMap API (free) | HIGH — next step |
| Local Events | Contextual (Events) | PredictHQ API | HIGH — catches demand spikes |

### 🔶 Needs Real Store Data
| Feature | Category | Data Source | Notes |
|---|---|---|---|
| Inventory Levels | Internal Store Data | POS system (Square/Clover) | Schema ready, needs integration |
| Price Point | Internal Store Data | POS system | Favorita has promo data only |
| Neighborhood Demographics | Contextual (Demographics) | Census API (free) | Mostly static, low priority for daily predictions. Better for new store onboarding / transfer learning |

### 🔴 Future / V2+
| Feature | Category | Data Source | Notes |
|---|---|---|---|
| Customer Proximity | Customer-Centric | PostGIS / Mobile App | Needs mobile app + user opt-in |
| Real-Time Demand | Customer-Centric | Live POS stream | Needs live POS integration |
| Trending Local Items | Customer-Centric | Aggregated user data | Network effect — needs many stores |

---

## OUTPUT FEATURES (What the system delivers to store owners)

### ✅ Built & Working
| Output | Description | Example | Status |
|---|---|---|---|
| Demand Forecast | Predicted units to sell | "Expect ~14,576 units of Beverages on Sunday" | ✅ XGBoost model |
| Contextual 'Why' | SHAP-based plain English explanation | "Because it's a Sunday, your 7-day avg is above normal, and this weekday averages 16,025 units" | ✅ SHAP translator |

### 🟡 Next To Build
| Output | Description | Example | What's Needed |
|---|---|---|---|
| Confidence Level | How certain the model is | "Prediction Confidence: 92% (High)" | Prediction intervals via quantile regression OR SHAP agreement score |
| Shortfall Warning | Gap between stock and predicted demand | "⚠️ Shortfall: You need 15 more units for Sunday" | Inventory data (POS integration or manual input) |
| Order Optimization | Exact reorder amount in supplier units | "Recommended Reorder: 2 Cases (24 units)" | Inventory + supplier pack size lookup table |
| Merchandising Tip (simple) | Which categories to feature prominently | "Beverages predicted to spike 40% — consider front placement" | Multi-category forecasts (almost ready) |

### 🔴 Future / V2+
| Output | Description | Example | What's Needed |
|---|---|---|---|
| Pricing Guidance | Dynamic pricing suggestions | "Demand is high; consider a 5% price increase" | Historical price data + price elasticity modeling. Hard to do well. Bad advice = lost trust. |
| Merchandising Tip (full) | Specific placement + timing | "Move Ponchos to front display by 2 PM" | Store layout data + time-of-day purchase patterns |

---

## MODEL PERFORMANCE BENCHMARKS

### Store 44 — Beverages
| Model | WMAPE | Notes |
|---|---|---|
| Baseline (4-week same-DOW avg) | 15.0% | The "dumb" prediction to beat |
| XGBoost (lag + calendar features) | 13.2% | 12.2% improvement over baseline |
| XGBoost + weather (not yet built) | ??? | Target: < 12% |
| XGBoost + weather + events (not yet built) | ??? | Target: < 10% |

### Store 44 — All Categories Average
| Model | WMAPE | Notes |
|---|---|---|
| Baseline | 12.3% | Ranges from 7.1% (Produce) to 19.5% (Personal Care) |
| XGBoost | Not yet tested across all categories | |

---

## TECH STACK
| Component | Technology | Status |
|---|---|---|
| ML Model | XGBoost (Tweedie regression) | ✅ Working |
| Explainability | SHAP (TreeExplainer) | ✅ Working |
| Feature Engineering | Python / Pandas | ✅ Working |
| Hyperparameter Tuning | Optuna | Built in config, not yet used |
| Database | PostgreSQL on Supabase (PostGIS) | Schema written, not yet deployed |
| Notifications | Twilio (WhatsApp/SMS) | Not yet built |
| Dashboard | Streamlit (planned) | Not yet built |
| Deployment | Supabase Edge Functions + Cloud Run (planned) | Not yet built |

---

## DEVELOPMENT PHILOSOPHY
- **Baseline first.** Every model must beat the simple average benchmark.
- **Temporal splits only.** Never random splits for time series data.
- **No future leakage.** Lag features must only look backward.
- **Explain everything.** A prediction without a "why" is useless to a store owner.
- **Throttle alerts.** Max 5 per store per day. Notification fatigue kills trust.
- **Minimize Human-in-the-Loop friction.** Meet owners where they are (WhatsApp/SMS), not where we wish they were (dashboards).

---

## ARCHITECTURE DECISIONS
| Decision | Choice | Why |
|---|---|---|
| Global model vs per-store | Global with store features | Not enough data per store to train individual models reliably |
| Loss function | Tweedie (not MSE) | Sales data is zero-inflated. Tweedie handles this. |
| Prediction granularity | Per-category per-day | Per-SKU is too noisy for small stores |
| Feature store | Single Postgres table | Training and inference read from same table. No skew. |
| Alert delivery | WhatsApp/SMS first | Store owners check phones, not dashboards |

---

## FILE STRUCTURE REFERENCE
```
feature_store/engineer.py    → build_features() and get_feature_columns()
model/evaluate.py            → compute_wmape()
config.py                    → All settings, thresholds, model params
db/schema.sql                → Full database schema (8 tables)
notebooks/01_eda.ipynb       → Data exploration
notebooks/02_baseline.ipynb  → Baseline model (WMAPE benchmarks)
notebooks/03_xgboost.ipynb   → XGBoost model training
notebooks/04_shap.ipynb      → SHAP explainability
```