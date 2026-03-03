"""
The Neighborhood Oracle — Dashboard
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="The Neighborhood Oracle",
    page_icon="🔮",
    layout="wide"
)

# ── Import our modules ───────────────────────────────────────────
from feature_store.engineer import build_features, get_feature_columns
from feature_store.sources.weather import generate_simulated_weather
from model.evaluate import compute_wmape


# ── Load Data (cached so it only runs once) ──────────────────────
@st.cache_data
def load_data():
    DATA_RAW = Path(r"C:\Users\syeds\OneDrive\Desktop\nboracle\nb_oracle\data\raw\store-sales-time-series-forecasting")
    train = pd.read_csv(DATA_RAW / "train.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_RAW / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_RAW / "stores.csv")
    weather = generate_simulated_weather(train["date"].sort_values().unique())
    return train, holidays, stores, weather


@st.cache_resource
def train_model(store_data, holidays, weather, category):
    """Train model for a specific store + category combo."""
    cat_data = (store_data[store_data.family == category]
                .sort_values("date").reset_index(drop=True))
    
    features = build_features(cat_data, holidays, weather_df=weather).dropna()
    feat_cols = get_feature_columns(features)
    
    SPLIT_DATE = "2017-07-15"
    train_df = features[features.index < SPLIT_DATE]
    test_df = features[features.index >= SPLIT_DATE]
    
    X_train = train_df[feat_cols]
    y_train = train_df["sales"]
    X_test = test_df[feat_cols]
    y_test = test_df["sales"]
    
    model = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.6,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        early_stopping_rounds=30)
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    predictions = np.maximum(model.predict(X_test), 0)
    wmape = compute_wmape(y_test.values, predictions)
    
    # Baseline
    baseline_pred = test_df["sales_same_dow_avg_4w"].fillna(0).values
    baseline_wmape = compute_wmape(y_test.values, baseline_pred)
    
    return model, X_test, y_test, predictions, feat_cols, wmape, baseline_wmape


def explain_day(model, features_row, feature_names, date):
    """Generate plain English explanation for one prediction."""
    explainer = shap.TreeExplainer(model)
    prediction = max(model.predict(features_row.values.reshape(1, -1))[0], 0)
    sv = explainer.shap_values(features_row.values.reshape(1, -1))[0]
    shap_series = pd.Series(sv, index=feature_names)
    top_3 = shap_series.abs().nlargest(3).index.tolist()

    reasons = []
    for feat in top_3:
        val = features_row[feat]
        shap_val = shap_series[feat]

        if feat == "is_weekend" and val == 1:
            reasons.append("it's a weekend (historically your busiest days)")
        elif feat == "is_weekend" and val == 0:
            reasons.append("it's a weekday (typically slower)")
        elif feat == "day_of_week":
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            reasons.append(f"it's a {day_names[int(val)]}")
        elif feat == "sales_avg_7d":
            direction = "above" if shap_val > 0 else "below"
            reasons.append(f"your recent 7-day average ({val:,.0f}) is {direction} normal")
        elif feat == "sales_1w_ago":
            reasons.append(f"same day last week you sold {val:,.0f} units")
        elif feat == "sales_same_dow_avg_4w":
            reasons.append(f"this weekday has been averaging {val:,.0f} units lately")
        elif feat == "sales_trend":
            direction = "trending up" if val > 0 else "trending down"
            reasons.append(f"sales have been {direction} this week")
        elif feat == "is_holiday" and val == 1:
            reasons.append("it's a holiday")
        elif feat == "is_payday" and val == 1:
            reasons.append("it's around payday (people spend more)")
        elif feat == "on_promotion" and val == 1:
            reasons.append("this category is on promotion")
        elif feat == "temp_delta_vs_yesterday":
            direction = "jumping up" if val > 0 else "dropping"
            reasons.append(f"temperature is {direction} by {abs(val):.0f}°F vs yesterday")
        elif feat == "is_precipitation":
            reasons.append("rain is expected — foot traffic may shift")
        elif feat == "is_hot_day" and val == 1:
            reasons.append("it's hotter than usual — cold items tend to spike")
        elif feat == "is_cold_day" and val == 1:
            reasons.append("it's cooler than usual — warm items may pick up")
        else:
            direction = "increasing" if shap_val > 0 else "decreasing"
            reasons.append(f"{feat.replace('_', ' ')} is {direction} demand")

    if len(reasons) >= 3:
        reasons_text = f"{reasons[0]}, {reasons[1]}, and {reasons[2]}"
    elif len(reasons) == 2:
        reasons_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reasons_text = reasons[0] if reasons else "historical patterns"

    return prediction, reasons_text, shap_series


# ── Load everything ──────────────────────────────────────────────
train, holidays, stores, weather = load_data()

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🔮 The Neighborhood Oracle")
st.sidebar.markdown("*Hyperlocal demand forecasting*")
st.sidebar.markdown("---")

# Store selector
store_numbers = sorted(train.store_nbr.unique())
selected_store = st.sidebar.selectbox("Select Store", store_numbers, index=store_numbers.index(44))

# Category selector
store_data = train[train.store_nbr == selected_store]
categories = sorted(store_data.family.unique())
selected_category = st.sidebar.selectbox("Select Category", categories, index=categories.index("BEVERAGES") if "BEVERAGES" in categories else 0)

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown(
    "1. Analyzes historical sales patterns\n"
    "2. Adds weather & calendar context\n"
    "3. Predicts future demand with XGBoost\n"
    "4. Explains *why* using SHAP"
)

# ── Train model ──────────────────────────────────────────────────
with st.spinner(f"Training model for Store {selected_store} — {selected_category}..."):
    model, X_test, y_test, predictions, feat_cols, wmape, baseline_wmape = train_model(
        store_data, holidays, weather, selected_category)

# ── Main Content ─────────────────────────────────────────────────
st.title("🔮 The Neighborhood Oracle")
st.markdown(f"### Store {selected_store} — {selected_category}")

# ── Key Metrics Row ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

improvement = (baseline_wmape - wmape) / baseline_wmape * 100

with col1:
    st.metric("Model Accuracy (WMAPE)", f"{wmape:.1%}")
with col2:
    st.metric("Baseline Accuracy", f"{baseline_wmape:.1%}")
with col3:
    st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
with col4:
    avg_daily = y_test.mean()
    st.metric("Avg Daily Sales", f"{avg_daily:,.0f} units")

st.markdown("---")

# ── Predictions Chart ────────────────────────────────────────────
st.subheader("📈 Predicted vs Actual Sales")

chart_data = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted": predictions
}, index=y_test.index)

st.line_chart(chart_data, height=400)

st.markdown("---")

# ── Daily Forecasts with Explanations ────────────────────────────
st.subheader("🔍 Daily Forecast Breakdown")
st.markdown("Select a date to see the prediction and its explanation.")

available_dates = y_test.index.tolist()
selected_date = st.select_slider(
    "Select Date",
    options=available_dates,
    value=available_dates[len(available_dates) // 2],
    format_func=lambda x: x.strftime("%b %d, %Y (%A)")
)

# Get the prediction for selected date
idx = available_dates.index(selected_date)
features_row = X_test.iloc[idx]
actual = y_test.values[idx]

prediction, reasons_text, shap_series = explain_day(
    model, features_row, feat_cols, selected_date)

# Display the alert
forecast_col, explain_col = st.columns([1, 1])

with forecast_col:
    st.markdown(f"#### 📅 {selected_date.strftime('%A, %B %d, %Y')}")
    
    delta = prediction - actual
    delta_pct = abs(delta) / actual * 100 if actual > 0 else 0
    
    st.metric("Predicted Demand", f"{prediction:,.0f} units",
              delta=f"{delta:+,.0f} vs actual ({delta_pct:.1f}% off)")
    st.metric("Actual Sales", f"{actual:,.0f} units")
    
    if delta_pct < 10:
        st.success("✅ Excellent prediction (within 10%)")
    elif delta_pct < 20:
        st.warning("⚠️ Decent prediction (within 20%)")
    else:
        st.error("❌ Off target (more than 20%)")

with explain_col:
    st.markdown("#### 🧠 Why This Prediction?")
    st.info(f"**Because** {reasons_text}.")
    
    # Show top SHAP features
    st.markdown("**Feature Influences:**")
    top_shap = shap_series.abs().nlargest(5)
    for feat in top_shap.index:
        val = features_row[feat]
        sv = shap_series[feat]
        icon = "🔺" if sv > 0 else "🔻"
        st.markdown(f"{icon} **{feat.replace('_', ' ').title()}** = {val:.1f} (impact: {sv:+.0f})")

st.markdown("---")

# ── Feature Importance ───────────────────────────────────────────
st.subheader("📊 What Drives Predictions?")

importance = pd.Series(
    model.feature_importances_,
    index=feat_cols
).sort_values(ascending=False).head(10)

st.bar_chart(importance)

st.markdown("---")

# ── Sample Alerts ────────────────────────────────────────────────
st.subheader("📱 Sample Store Owner Alerts")
st.markdown("*This is what the store owner would receive via WhatsApp/SMS:*")

# Generate alerts for the last 5 days
for i in range(-5, 0):
    idx = i
    date = available_dates[idx]
    features_row = X_test.iloc[idx]
    actual = y_test.values[idx]
    
    pred, reasons, _ = explain_day(model, features_row, feat_cols, date)
    
    shortfall = actual - pred
    if shortfall > actual * 0.15:
        emoji = "🔴"
        status = "UNDERSTOCKED"
    elif shortfall < -actual * 0.15:
        emoji = "🟡"
        status = "OVERSTOCKED"
    else:
        emoji = "🟢"
        status = "ON TRACK"
    
    with st.expander(f"{emoji} {date.strftime('%A, %b %d')} — {status} — Predicted: {pred:,.0f} | Actual: {actual:,.0f}"):
        st.markdown(f"**Forecast:** ~{pred:,.0f} units of {selected_category}")
        st.markdown(f"**Why:** {reasons}")
        st.markdown(f"**What happened:** {actual:,.0f} units actually sold")
        
        diff_pct = abs(pred - actual) / actual * 100 if actual > 0 else 0
        st.markdown(f"**Accuracy:** {100 - diff_pct:.1f}%")

st.markdown("---")
st.caption("Built with 🔮 The Neighborhood Oracle | XGBoost + SHAP | Powered by Streamlit")