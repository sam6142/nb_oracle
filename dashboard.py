"""
The Neighborhood Oracle — Dashboard v3 (Boston + Live Weather + Events + Optuna)
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from feature_store.engineer import build_features, get_feature_columns
from feature_store.sources.weather import fetch_weather_forecast
from feature_store.sources.events import generate_boston_events, events_to_features
from model.evaluate import compute_wmape
from explainability.translator import get_shap_explanation, format_whatsapp_message
from model.train import load_model, train_and_save
from config import get_dataset_path, get_model_params, MODEL_REGISTRY
from model.train import load_model as load_saved_model, train_and_save

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="The Neighborhood Oracle",
    page_icon="🔮",
    layout="wide"
)

# ── Data Paths (from config.py — no hardcoded paths) ─────────────
BOSTON_DATA = get_dataset_path()
SPLIT_DAYS_BACK = 30
BEST_PARAMS = get_model_params()


# ── Load Data ────────────────────────────────────────────────────
@st.cache_data
def load_boston_data():
    train = pd.read_csv(BOSTON_DATA / "train.csv", parse_dates=["date"])
    holidays = pd.read_csv(BOSTON_DATA / "holidays_events.csv", parse_dates=["date"])
    weather = pd.read_csv(BOSTON_DATA / "weather_pipeline.csv", parse_dates=["date"])
    
    events_path = BOSTON_DATA / "events_features.csv"
    if events_path.exists():
        events = pd.read_csv(events_path, parse_dates=["date"])
    else:
        start = train["date"].min().strftime("%Y-%m-%d")
        end = train["date"].max().strftime("%Y-%m-%d")
        raw_events = generate_boston_events(start, end, seed=42)
        events = events_to_features(raw_events, pd.date_range(start, end))
    
    return train, holidays, weather, events


@st.cache_data(ttl=3600)
def load_live_forecast():
    # Returns None if the Open-Meteo API is unreachable (SSL/network issue).
    # The dashboard handles None gracefully — weather sections are skipped.
    try:
        return fetch_weather_forecast()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_upcoming_events():
    today = datetime.now().strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    raw = generate_boston_events(today, next_week,
                                 seed=int(datetime.now().timestamp()) % 10000)
    dates = pd.date_range(today, next_week)
    features = events_to_features(raw, dates)
    return raw, features


@st.cache_resource
@st.cache_resource
def train_model(_train, _holidays, _weather, _events, category):
    """Load a saved model, or train one if none exists."""
    
    # Try loading a saved model first
    model, metadata = load_model(category)
    
    if model is not None:
        # Rebuild test data for display purposes
        cat_data = (_train[_train.family == category]
                    .sort_values("date").reset_index(drop=True))
        
        features = build_features(cat_data, _holidays,
                                   weather_df=_weather, events_df=_events).dropna()
        feat_cols = metadata["feature_columns"]
        
        split = metadata.get("split_date", 
                (datetime.now() - timedelta(days=SPLIT_DAYS_BACK + 1)).strftime("%Y-%m-%d"))
        
        test_df = features[features.index >= split]
        
        if len(test_df) < 1:
            return None
        
        # Only use features the model knows about
        available_cols = [c for c in feat_cols if c in test_df.columns]
        X_test = test_df[available_cols]
        y_test = test_df["sales"]
        
        predictions = np.maximum(model.predict(X_test), 0)
        wmape = compute_wmape(y_test.values, predictions)
        
        baseline_pred = test_df["sales_same_dow_avg_4w"].fillna(0).values
        baseline_wmape = compute_wmape(y_test.values, baseline_pred)
        
        return {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "predictions": predictions,
            "feat_cols": available_cols,
            "wmape": wmape,
            "baseline_wmape": baseline_wmape,
            "cat_data": cat_data,
        }
    
    # No saved model — train one on the fly and save it
    result = train_and_save(category)
    if result is None:
        return None
    
    cat_data = (_train[_train.family == category]
                .sort_values("date").reset_index(drop=True))
    
    return {
        "model": result["model"],
        "X_test": result["X_test"],
        "y_test": result["y_test"],
        "predictions": result["predictions"],
        "feat_cols": result["feat_cols"],
        "wmape": result["metadata"]["wmape"],
        "baseline_wmape": result["metadata"]["baseline_wmape"],
        "cat_data": cat_data,
    }


def make_live_prediction(model, feat_cols, recent_sales, holidays,
                          weather_hist, forecast_day, event_row=None):
    forecast_date = pd.Timestamp(forecast_day["date"])
    
    future_row = pd.DataFrame({
        "date": [forecast_date],
        "sales": [0],
        "onpromotion": [0],
    })
    
    combined = pd.concat([
        recent_sales[["date", "sales", "onpromotion"]].tail(60),
        future_row
    ]).reset_index(drop=True)
    
    combined_dates = pd.to_datetime(combined["date"])
    weather_for_combined = weather_hist[weather_hist["date"].isin(combined_dates)].copy()
    
    weather_row = pd.DataFrame({
        "date": [forecast_date],
        "temp_high": [forecast_day["temp_high"]],
        "temp_feels_like": [forecast_day["temp_feels_like"]],
        "precipitation_mm": [forecast_day["precipitation_mm"]],
        "is_precipitation": [forecast_day["is_precipitation"]],
        "humidity": [65],
    })
    
    weather_for_combined = pd.concat([weather_for_combined, weather_row]).drop_duplicates(subset="date")
    
    events_for_combined = None
    if event_row is not None:
        all_dates = combined_dates.tolist()
        empty_events = pd.DataFrame({
            "date": all_dates[:-1],
            "nearby_events": 0, "nearby_attendance": 0,
            "city_events": 0, "city_attendance": 0,
            "has_sports_nearby": 0, "has_sports_city": 0,
            "has_music": 0, "is_marathon": 0, "event_score": 0,
        })
        forecast_event = pd.DataFrame([event_row])
        forecast_event["date"] = forecast_date
        events_for_combined = pd.concat([empty_events, forecast_event]).reset_index(drop=True)
        events_for_combined["date"] = pd.to_datetime(events_for_combined["date"])
    
    features = build_features(combined, holidays,
                               weather_df=weather_for_combined,
                               events_df=events_for_combined).dropna()
    
    if len(features) == 0:
        return None, None
    
    last_row = features.iloc[-1]
    
    feature_row = pd.Series(0.0, index=feat_cols)
    for col in feat_cols:
        if col in last_row.index:
            feature_row[col] = last_row[col]
    
    prediction = max(model.predict(feature_row.values.reshape(1, -1))[0], 0)
    result = get_shap_explanation(model, feature_row, feat_cols, forecast_date, prediction)
    
    return prediction, result


def explain_day_historical(model, features_row, feature_names, date):
    prediction = max(model.predict(features_row.values.reshape(1, -1))[0], 0)
    result = get_shap_explanation(model, features_row, feature_names, date, prediction)
    return prediction, result["summary"], result["explanation"], result["confidence"], result["shap_series"]


# ── Load everything ──────────────────────────────────────────────
train, holidays, weather, events = load_boston_data()
live_forecast = load_live_forecast()
raw_upcoming_events, upcoming_event_features = load_upcoming_events()


# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🔮 The Neighborhood Oracle")
st.sidebar.markdown("*Dorchester Bodega — Live Demo*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Live Forecast", "📊 Store Dashboard", "📱 Alert Preview"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources:**")
st.sidebar.markdown(
    "☀️ Weather: Open-Meteo (live)\n"
    "🎫 Events: Boston area\n"
    "📊 Sales: Mock bodega data\n"
    "🤖 Model: XGBoost + Optuna + SHAP"
)

optuna_path = MODEL_REGISTRY / "optuna_best_params.json"
if optuna_path.exists():
    st.sidebar.success("✅ Using Optuna-tuned params")
else:
    st.sidebar.warning("⚠️ Using default params")


# ═══════════════════════════════════════════════════════════════
# PAGE 1: LIVE FORECAST
# ═══════════════════════════════════════════════════════════════
if page == "🔮 Live Forecast":
    
    st.title("🔮 Live 7-Day Forecast")
    st.markdown("Real weather + Boston events → demand predictions for the week ahead")
    
    categories = sorted(train.family.unique())
    default_cat = categories.index("BEVERAGES") if "BEVERAGES" in categories else 0
    selected_category = st.selectbox("Select Category", categories, index=default_cat)
    
    with st.spinner(f"Training model for {selected_category}..."):
        result = train_model(train, holidays, weather, events, selected_category)
    
    if result is None:
        st.warning("Not enough data for this category.")
    else:
        model = result["model"]
        feat_cols = result["feat_cols"]
        cat_data = result["cat_data"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy (WMAPE)", f"{result['wmape']:.1%}")
        with col2:
            st.metric("Baseline Accuracy", f"{result['baseline_wmape']:.1%}")
        with col3:
            imp = (result['baseline_wmape'] - result['wmape']) / result['baseline_wmape'] * 100
            st.metric("Improvement", f"{imp:.1f}%", delta=f"{imp:+.1f}%")
        
        st.markdown("---")
        
        st.subheader("🌤️ This Week's Weather + Events")
        
        if live_forecast is None:
            st.warning("Live weather unavailable (network/SSL error). Forecast will run without real weather data.")
        if live_forecast is not None:
            weather_cols = st.columns(7)
            for i, (_, fday) in enumerate(live_forecast.iterrows()):
                if i >= 7:
                    break
                dt = pd.Timestamp(fday["date"])
                rain = "🌧️" if fday["is_precipitation"] else "☀️"
                
                day_events = raw_upcoming_events[
                    raw_upcoming_events["date"] == dt.strftime("%Y-%m-%d")
                ] if not raw_upcoming_events.empty else pd.DataFrame()
                
                with weather_cols[i]:
                    st.markdown(f"**{dt.strftime('%a')}**")
                    st.markdown(f"## {rain}")
                    st.markdown(f"**{fday['temp_high']:.0f}°F**")
                    if not day_events.empty:
                        for _, ev in day_events.head(2).iterrows():
                            st.caption(f"🎫 {ev['name'][:20]}")
        
        st.markdown("---")
        
        st.subheader(f"📦 7-Day {selected_category} Forecast")
        
        if live_forecast is not None:
            forecast_data = []
            
            for _, fday in live_forecast.iterrows():
                forecast_date = pd.Timestamp(fday["date"])
                
                event_row = None
                if not upcoming_event_features.empty:
                    match = upcoming_event_features[
                        pd.to_datetime(upcoming_event_features["date"]) == forecast_date
                    ]
                    if not match.empty:
                        event_row = match.iloc[0].to_dict()
                
                pred, expl = make_live_prediction(
                    model, feat_cols, cat_data, holidays, weather, fday, event_row)
                
                if pred is not None:
                    day_events = raw_upcoming_events[
                        raw_upcoming_events["date"] == forecast_date.strftime("%Y-%m-%d")
                    ] if not raw_upcoming_events.empty else pd.DataFrame()
                    
                    event_names = ", ".join(day_events["name"].values[:2]) if not day_events.empty else "None"
                    
                    forecast_data.append({
                        "date": forecast_date,
                        "prediction": pred,
                        "summary": expl["summary"],
                        "explanation": expl["explanation"],
                        "confidence": expl["confidence"],
                        "temp": fday["temp_high"],
                        "rain": fday["is_precipitation"],
                        "events": event_names,
                    })
            
            for fd in forecast_data:
                conf_emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
                rain_icon = "🌧️" if fd["rain"] else "☀️"
                
                with st.expander(
                    f"📅 {fd['date'].strftime('%A, %B %d')}  |  "
                    f"**~{fd['prediction']:.0f} units**  |  "
                    f"{rain_icon} {fd['temp']:.0f}°F  |  "
                    f"{conf_emoji.get(fd['confidence'], '⚪')} {fd['confidence'].capitalize()}"
                ):
                    st.markdown(f"**{fd['summary']}**")
                    st.info(fd["explanation"])
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Predicted", f"{fd['prediction']:.0f} units")
                    with c2:
                        st.metric("Temperature", f"{fd['temp']:.0f}°F")
                    with c3:
                        st.metric("Events", fd["events"][:30])
        
        st.markdown("---")
        
        st.subheader("📊 What Drives Predictions?")
        importance = pd.Series(
            model.feature_importances_, index=feat_cols
        ).sort_values(ascending=False).head(15)
        st.bar_chart(importance)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: STORE DASHBOARD (Historical)
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Store Dashboard":
    
    st.title("📊 Historical Performance")
    st.markdown("How accurate has the model been on past data?")
    
    categories = sorted(train.family.unique())
    default_cat = categories.index("BEVERAGES") if "BEVERAGES" in categories else 0
    selected_category = st.selectbox("Select Category", categories, index=default_cat)
    
    with st.spinner(f"Training model for {selected_category}..."):
        result = train_model(train, holidays, weather, events, selected_category)
    
    if result is None:
        st.warning("Not enough data for this category.")
    else:
        model = result["model"]
        X_test = result["X_test"]
        y_test = result["y_test"]
        predictions = result["predictions"]
        feat_cols = result["feat_cols"]
        wmape = result["wmape"]
        baseline_wmape = result["baseline_wmape"]
        
        improvement = (baseline_wmape - wmape) / baseline_wmape * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Error (WMAPE)", f"{wmape:.1%}")
        with col2:
            st.metric("Baseline Error", f"{baseline_wmape:.1%}")
        with col3:
            st.metric("Improvement", f"{improvement:.1f}%",
                      delta=f"{improvement:+.1f}%")
        with col4:
            st.metric("Avg Daily Sales", f"{y_test.mean():,.0f}")
        
        st.markdown("---")
        
        st.subheader("📈 Predicted vs Actual (Last 30 Days)")
        chart_data = pd.DataFrame({
            "Actual Sales": y_test.values,
            "Predicted": predictions
        }, index=y_test.index)
        st.line_chart(chart_data, height=350)
        
        st.markdown("---")
        
        st.subheader("🔍 Daily Breakdown")
        
        available_dates = y_test.index.tolist()
        selected_date = st.select_slider(
            "Select Date",
            options=available_dates,
            value=available_dates[len(available_dates) // 2],
            format_func=lambda x: x.strftime("%b %d, %Y (%A)")
        )
        
        idx = available_dates.index(selected_date)
        features_row = X_test.iloc[idx]
        actual = y_test.values[idx]
        
        prediction, summary, explanation, confidence, shap_series = explain_day_historical(
            model, features_row, feat_cols, selected_date)
        
        fc_col, ex_col = st.columns([1, 1])
        
        with fc_col:
            st.markdown(f"#### 📅 {selected_date.strftime('%A, %B %d, %Y')}")
            delta_val = prediction - actual
            if actual > 0:
                delta_pct = abs(delta_val) / actual * 100
            elif prediction > 0:
                delta_pct = 100  # Predicted something but actual was 0 = bad
            else:
                delta_pct = 0  # Both zero = fine
            
            st.metric("Predicted", f"{prediction:,.0f} units",
                      delta=f"{delta_val:+,.0f} vs actual ({delta_pct:.1f}% off)")
            st.metric("Actual", f"{actual:,.0f} units")
            
            if delta_pct < 10:
                st.success("✅ Excellent prediction (within 10%)")
            elif delta_pct < 20:
                st.warning("⚠️ Decent prediction (within 20%)")
            else:
                st.error("❌ Off target (more than 20%)")
        
        with ex_col:
            st.markdown("#### 🧠 Why This Prediction?")
            conf_emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
            st.markdown(f"**{summary}**")
            st.info(explanation)
            st.markdown(f"{conf_emoji.get(confidence, '⚪')} Confidence: **{confidence.capitalize()}**")


# ═══════════════════════════════════════════════════════════════
# PAGE 3: ALERT PREVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "📱 Alert Preview":
    
    st.title("📱 WhatsApp Alert Preview")
    st.markdown("What the store owner would see on their phone tomorrow morning.")
    
    categories = sorted(train.family.unique())
    default_cat = categories.index("BEVERAGES") if "BEVERAGES" in categories else 0
    selected_category = st.selectbox("Category", categories, index=default_cat)
    
    with st.spinner("Training model..."):
        result = train_model(train, holidays, weather, events, selected_category)
    
    if result is None:
        st.warning("Not enough data.")
    else:
        model = result["model"]
        feat_cols = result["feat_cols"]
        cat_data = result["cat_data"]
        
        st.markdown("---")
        
        if live_forecast is not None and len(live_forecast) > 1:
            st.subheader("📲 Tomorrow's Alert")
            
            tomorrow = live_forecast.iloc[1]
            tomorrow_date = pd.Timestamp(tomorrow["date"])
            
            event_row = None
            if not upcoming_event_features.empty:
                match = upcoming_event_features[
                    pd.to_datetime(upcoming_event_features["date"]) == tomorrow_date
                ]
                if not match.empty:
                    event_row = match.iloc[0].to_dict()
            
            pred, expl = make_live_prediction(
                model, feat_cols, cat_data, holidays, weather, tomorrow, event_row)
            
            if pred is not None:
                fake_inventory = int(pred * np.random.uniform(0.7, 1.1))
                
                msg = format_whatsapp_message(
                    store_name="Dorchester Bodega",
                    category=selected_category,
                    date=tomorrow_date,
                    prediction=pred,
                    explanation_result=expl,
                    current_inventory=fake_inventory,
                )
                
                st.markdown(
                    f"""
                    <div style="
                        max-width: 400px;
                        margin: 20px auto;
                        padding: 20px;
                        background: linear-gradient(135deg, #075e54, #128c7e);
                        border-radius: 15px;
                        color: white;
                        font-family: -apple-system, sans-serif;
                        font-size: 14px;
                        line-height: 1.6;
                        white-space: pre-wrap;
                    ">
{msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                
                st.subheader("📅 Full Week Preview")
                
                for _, fday in live_forecast.iterrows():
                    fdate = pd.Timestamp(fday["date"])
                    
                    ev_row = None
                    if not upcoming_event_features.empty:
                        match = upcoming_event_features[
                            pd.to_datetime(upcoming_event_features["date"]) == fdate
                        ]
                        if not match.empty:
                            ev_row = match.iloc[0].to_dict()
                    
                    p, e = make_live_prediction(
                        model, feat_cols, cat_data, holidays, weather, fday, ev_row)
                    
                    if p is not None:
                        rain = "🌧️" if fday["is_precipitation"] else "☀️"
                        conf_emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
                        
                        day_events = raw_upcoming_events[
                            raw_upcoming_events["date"] == fdate.strftime("%Y-%m-%d")
                        ] if not raw_upcoming_events.empty else pd.DataFrame()
                        
                        event_str = ""
                        if not day_events.empty:
                            event_str = f" | 🎫 {', '.join(day_events['name'].values[:2])}"
                        
                        with st.expander(
                            f"{fdate.strftime('%A %b %d')} | "
                            f"~{p:.0f} units | "
                            f"{rain} {fday['temp_high']:.0f}°F"
                            f"{event_str}"
                        ):
                            st.markdown(f"**{e['summary']}**")
                            st.info(e["explanation"])
                            st.markdown(f"{conf_emoji.get(e['confidence'], '⚪')} Confidence: **{e['confidence'].capitalize()}**")
        
        st.markdown("---")
        
        st.subheader("⚙️ Alert Settings")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.slider("Max alerts per day", 1, 10, 5, key="max_alerts")
        with s2:
            st.slider("Spike threshold (%)", 10, 50, 30, key="spike_thresh")
        with s3:
            st.selectbox("Delivery channel", ["WhatsApp", "SMS", "Both"], key="channel")


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🔮 The Neighborhood Oracle | Dorchester, MA | "
    "Live weather from Open-Meteo | XGBoost + Optuna + SHAP | "
    f"Last updated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}"
)