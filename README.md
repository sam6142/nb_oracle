# 🔮 The Neighborhood Oracle

**Hyperlocal demand forecasting for small businesses.**

Predicts what a store will sell — and *why* — by combining historical sales data with real-time context like weather and local events. Delivers alerts via WhatsApp/SMS so store owners can stock smarter.

## Quick Start

```bash
git clone https://github.com/sam6142/nb_oracle.git
cd nb_oracle
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## How It Works

```
Sales History ──┐
Weather Data  ──┤──→ Feature Engineering ──→ XGBoost Model ──→ "Stock up on
Local Events  ──┤                                                beverages by
Calendar      ──┘                                                Thursday" → SMS
```

## Status

- [x] Project structure
- [ ] Data exploration
- [ ] Baseline model
- [ ] XGBoost training pipeline
- [ ] SHAP explainability
- [ ] Weather & events integration
- [ ] WhatsApp/SMS alerts
- [ ] Dashboard