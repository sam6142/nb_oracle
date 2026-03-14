# Product Direction — The Neighborhood Oracle

## Last Updated: March 4, 2026

---

## Core Philosophy

**Predictions are the foundation. Intelligence is the product.**

Don't compete with the owner's gut on obvious patterns (weekends are busy, storms need bread). His 20 years of experience beats our 14 months of data on those.

Instead, provide value through things he CAN'T know on his own:
- Cultural calendars outside his personal experience
- EBT/SNAP disbursement cycle quantification
- Demographic-specific demand patterns
- Cross-neighborhood intelligence
- "Why did sales drop?" diagnostics

**The shift:** From "predicted demand: 120 units" (data he can guess) to "you're leaving $180 on the table this EBT cycle because you run out of rice by noon" (specific advice with a dollar sign he can't calculate).

---

## Two Product Layers

### Layer 1: Prediction Engine (Built ✅)
The XGBoost pipeline that predicts demand per category per day. This is infrastructure — it powers everything but is NOT the product by itself.

- Historical sales → lag features → calendar → weather → events → XGBoost → predicted units
- SHAP for explainability
- WMAPE for accuracy tracking

### Layer 2: Intelligence Layer (Building 🔨)
The insights, diagnostics, and recommendations that turn predictions into business advice.

| Feature | What It Does | Value Level |
|---|---|---|
| **Sales Drop Diagnostic** | "Why was last week bad?" — uses SHAP to explain past performance drops | 🔥🔥🔥 Highest |
| **EBT/SNAP Cycle Alerts** | Quantifies the payment cycle effect per category with $ estimates | 🔥🔥🔥 High |
| **Cultural Calendar** | Alerts for holidays outside owner's personal experience with category-specific prep lists | 🔥🔥🔥 High |
| **Spoilage Risk Warnings** | "Cut produce order 30% this week — 95°F means waste doubles" | 🔥🔥 Medium-High |
| **Daily Summary Message** | One message covering all categories, not 33 separate alerts | 🔥🔥 Medium-High |
| **Storm/Extreme Weather Playbook** | Pre-storm → during → after timeline with category actions | 🔥🔥 Medium |
| **Cross-Category Correlations** | "Diaper sales up → stock more baby food" | 🔥 Medium |
| **Price Sensitivity Alerts** | "Egg prices up 22%, customers switching — stock alternatives" | 🔥 Medium (needs price data) |
| **New Product Recommendations** | "Stores like yours also carry X — potential $Y/month" | 🔥 Future |
| **Competitor Intelligence** | "New grocery opened 4 blocks away — expect 10-15% traffic shift" | 🔥 Future |

---

## The Diagnostic Use Case: "Why Did Sales Drop?"

This is the #1 priority for the intelligence layer.

### The Problem
Owner notices revenue dropped 15% last month. He doesn't know why. Was it:
- Weather? (Bad month of rain)
- A competitor? (New store opened nearby)
- Seasonal? (Normal January dip)
- Demographic? (Neighborhood change)
- One category dragging everything down?
- Or several small things compounding?

His gut says "it was slow" but can't break it down.

### What The Oracle Would Say

```
📊 Monthly Report — January 2026

Revenue: $48,200 (down 14.2% vs December)

Here's what happened:

1. SEASONAL DIP accounts for ~8% of the drop
   January is historically your slowest month. This is normal
   and matches what we see at similar stores.

2. WEATHER added another ~4%
   January had 6 snow days (vs 2 last year). Each snow day
   reduces your foot traffic by ~40%. That's ~$1,800 in lost
   revenue just from weather.

3. BEVERAGES dropped 22% — more than seasonal
   This is unusual. Cold beverages always dip in winter, but
   22% is steeper than expected. Possible causes:
   • Price increase on your top-selling water brand?
   • New competitor carrying similar products?

4. DAIRY held strong (+2% vs December)
   Your dairy section is outperforming. Whatever you're doing
   there, keep doing it.

💡 Recommendation: The seasonal and weather drops are temporary.
   Investigate the beverage decline — that's the one thing
   you can act on.
```

### How We Build It
- Compare actual sales vs predicted sales for the past period
- Use SHAP to identify which features caused the biggest deviations
- Group explanations into categories (weather, seasonal, category-specific)
- Quantify each factor in dollar terms
- Generate the narrative

### Technical Approach
1. Run predictions for every day in the analysis period
2. Calculate residuals (actual - predicted) per day per category
3. For days with large residuals, use SHAP to explain why
4. Aggregate SHAP explanations across the period
5. Cluster into themes (weather days, event days, unexplained drops)
6. Compose the narrative using templates

---

## What Makes This NOT a Gimmick

| Gimmick (avoid) | Real Value (build this) |
|---|---|
| "Saturday will be busy" | "You leave $180 on the table every EBT cycle" |
| "Stock up for the storm" | "Cut produce 30%, double water, skip Thursday delivery" |
| "Predicted demand: 120" | "Beverages dropped 22% — steeper than seasonal. Investigate." |
| "Confidence: High" | "6 snow days cost you $1,800 this month" |
| Competing with the owner's gut | Telling him things his gut can't know |

---

## Build Order

1. ✅ Prediction engine (XGBoost + SHAP + weather + events)
2. 🔨 Sales drop diagnostic ("why was last week bad?")
3. 📋 Daily summary message (one message, all categories, action list)
4. 📋 EBT/SNAP cycle quantification with $ estimates
5. 📋 Cultural calendar integration
6. 📋 Spoilage risk model
7. 📋 Storm/extreme weather playbook
8. 📋 Price sensitivity (needs external data)

---

*This document guides all product decisions. When in doubt, ask: "does this tell the owner something he doesn't already know?"*