# Neighborhood Oracle

Hyperlocal demand forecasting for small businesses (bodegas, small grocery stores).

## What This Project Does
Predicts how much of each product category a store will sell in the next 1-7 days,
using historical sales + weather + local events. Sends WhatsApp/SMS alerts to store
owners when they need to restock.

## Architecture
- Feature store pattern: all features go into one Postgres table called `feature_vectors`
- Model: XGBoost with Tweedie regression for zero-inflated sales data
- Explainability: SHAP values translated to plain English
- Alerts: WhatsApp/SMS via Twilio

## Rules for Code in This Project
- All config in config.py — no magic numbers in other files
- Time series data: NEVER use random train/test splits, always split by date
- Lag features must only look backward (no future data leakage)
- Predictions must be non-negative: always use np.maximum(pred, 0)