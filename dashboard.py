"""
The Neighborhood Oracle — Dashboard v2
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from feature_store.engineer import build_features, get_feature_columns
from feature_store.sources.weather import generate_simulated_weather
from model.evaluate import compute_wmape

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="The Neighborhood Oracle",
    page_icon="🔮",
    layout="wide"
)

# ── Load Data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    DATA_RAW = Path(r"C:\Users\syeds\OneDrive\Desktop\nboracle\nb_oracle\data\raw\store-sales-time-series-forecasting")
    train = pd.read_csv(DATA_RAW / "train.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_RAW / "holidays_events.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_RAW / "stores.csv")
    weather = generate_simulated_weather(train["date"].sort_values().unique())
    return train, holidays, stores, weather

SPLIT_DATE = "2017-07-15"

@st.cache_resource
def train_model(store_data, holidays, weather, category):
    cat_data = (store_data[store_data.family == category]
                .sort_values("date").reset_index(drop=True))
    
    if len(cat_data) < 100 or cat_data["sales"].sum() < 50:
        return None
    
    features = build_features(cat_data, holidays, weather_df=weather).dropna()
    feat_cols = get_feature_columns(features)
    
    train_df = features[features.index < SPLIT_DATE]
    test_df = features[features.index >= SPLIT_DATE]
    
    if len(train_df) < 50 or len(test_df) < 5 or test_df["sales"].sum() == 0:
        return None
    
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
    
    baseline_pred = test_df["sales_same_dow_avg_4w"].fillna(0).values
    baseline_wmape = compute_wmape(y_test.values, baseline_pred)
    
    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "feat_cols": feat_cols,
        "wmape": wmape,
        "baseline_wmape": baseline_wmape,
    }


@st.cache_data
def run_multi_store_analysis(_train, _holidays, _weather, stores_list, categories_list):
    """Run model across multiple stores and categories."""
    results = []
    for store_num in stores_list:
        store_data = _train[_train.store_nbr == store_num]
        for cat in categories_list:
            cat_data = (store_data[store_data.family == cat]
                        .sort_values("date").reset_index(drop=True))
            
            if len(cat_data) < 100 or cat_data["sales"].sum() < 50:
                continue
            
            try:
                features = build_features(cat_data, _holidays, weather_df=_weather).dropna()
                feat_cols = get_feature_columns(features)
                
                train_df = features[features.index < SPLIT_DATE]
                test_df = features[features.index >= SPLIT_DATE]
                
                if len(train_df) < 50 or len(test_df) < 5 or test_df["sales"].sum() == 0:
                    continue
                
                X_train, y_train = train_df[feat_cols], train_df["sales"]
                X_test, y_test = test_df[feat_cols], test_df["sales"]
                
                model = xgb.XGBRegressor(
                    objective="reg:tweedie", tweedie_variance_power=1.6,
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                    early_stopping_rounds=30)
                
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                preds = np.maximum(model.predict(X_test), 0)
                
                baseline_pred = test_df["sales_same_dow_avg_4w"].fillna(0).values
                
                results.append({
                    "store": store_num,
                    "category": cat,
                    "xgb_wmape": compute_wmape(y_test.values, preds),
                    "baseline_wmape": compute_wmape(y_test.values, baseline_pred),
                    "avg_daily_sales": y_test.mean(),
                    "beats_baseline": compute_wmape(y_test.values, preds) < compute_wmape(y_test.values, baseline_pred),
                })
            except:
                continue
    
    return pd.DataFrame(results)


def explain_day(model, features_row, feature_names, date):
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


# ── Load data ────────────────────────────────────────────────────
train, holidays, stores_info, weather = load_data()

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🔮 The Neighborhood Oracle")
st.sidebar.markdown("*Hyperlocal demand forecasting*")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["📊 Store Dashboard", "🏪 Multi-Store Overview", "📱 Alert Preview"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown(
    "1. Analyzes historical sales\n"
    "2. Adds weather & calendar context\n"
    "3. Predicts demand with XGBoost\n"
    "4. Explains *why* using SHAP"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Stats:**")
st.sidebar.markdown("• 88% category win rate\n• 11.6% avg WMAPE\n• 50+ store-category combos tested")


# ═══════════════════════════════════════════════════════════════
# PAGE 1: STORE DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "📊 Store Dashboard":
    
    st.title("🔮 Store Dashboard")
    
    # Store and category selectors in columns
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        store_numbers = sorted(train.store_nbr.unique())
        selected_store = st.selectbox("Select Store", store_numbers, 
                                       index=store_numbers.index(44))
    with sel_col2:
        store_data = train[train.store_nbr == selected_store]
        categories = sorted(store_data.family.unique())
        default_cat = categories.index("BEVERAGES") if "BEVERAGES" in categories else 0
        selected_category = st.selectbox("Select Category", categories, index=default_cat)
    
    # Train model
    with st.spinner(f"Training model for Store {selected_store} — {selected_category}..."):
        result = train_model(store_data, holidays, weather, selected_category)
    
    if result is None:
        st.warning(f"Not enough data for {selected_category} at Store {selected_store}.")
    else:
        model = result["model"]
        X_test = result["X_test"]
        y_test = result["y_test"]
        predictions = result["predictions"]
        feat_cols = result["feat_cols"]
        wmape = result["wmape"]
        baseline_wmape = result["baseline_wmape"]
        
        improvement = (baseline_wmape - wmape) / baseline_wmape * 100
        
        # ── Metrics Row ──────────────────────────────────────
        st.markdown(f"### Store {selected_store} — {selected_category}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Error (WMAPE)", f"{wmape:.1%}")
        with col2:
            st.metric("Baseline Error", f"{baseline_wmape:.1%}")
        with col3:
            delta_color = "normal" if improvement > 0 else "inverse"
            st.metric("Improvement", f"{improvement:.1f}%", 
                      delta=f"{improvement:+.1f}%", delta_color=delta_color)
        with col4:
            st.metric("Avg Daily Sales", f"{y_test.mean():,.0f}")
        
        st.markdown("---")
        
        # ── Predictions Chart ────────────────────────────────
        st.subheader("📈 Predicted vs Actual Sales")
        chart_data = pd.DataFrame({
            "Actual Sales": y_test.values,
            "Predicted": predictions
        }, index=y_test.index)
        st.line_chart(chart_data, height=350)
        
        st.markdown("---")
        
        # ── Daily Forecast Breakdown ─────────────────────────
        st.subheader("🔍 Daily Forecast Breakdown")
        
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
        
        prediction, reasons_text, shap_series = explain_day(
            model, features_row, feat_cols, selected_date)
        
        fc_col, ex_col = st.columns([1, 1])
        
        with fc_col:
            st.markdown(f"#### 📅 {selected_date.strftime('%A, %B %d, %Y')}")
            
            delta_val = prediction - actual
            delta_pct = abs(delta_val) / actual * 100 if actual > 0 else 0
            
            st.metric("Predicted Demand", f"{prediction:,.0f} units",
                      delta=f"{delta_val:+,.0f} vs actual ({delta_pct:.1f}% off)")
            st.metric("Actual Sales", f"{actual:,.0f} units")
            
            if delta_pct < 10:
                st.success("✅ Excellent prediction (within 10%)")
            elif delta_pct < 20:
                st.warning("⚠️ Decent prediction (within 20%)")
            else:
                st.error("❌ Off target (more than 20%)")
        
        with ex_col:
            st.markdown("#### 🧠 Why This Prediction?")
            st.info(f"**Because** {reasons_text}.")
            
            st.markdown("**Top Feature Influences:**")
            top_shap = shap_series.abs().nlargest(5)
            for feat in top_shap.index:
                val = features_row[feat]
                sv = shap_series[feat]
                icon = "🔺" if sv > 0 else "🔻"
                st.markdown(f"{icon} **{feat.replace('_', ' ').title()}** = {val:.1f}")
        
        st.markdown("---")
        
        # ── Feature Importance ───────────────────────────────
        st.subheader("📊 What Drives Predictions?")
        importance = pd.Series(
            model.feature_importances_, index=feat_cols
        ).sort_values(ascending=False).head(10)
        st.bar_chart(importance)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: MULTI-STORE OVERVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "🏪 Multi-Store Overview":
    
    st.title("🏪 Multi-Store Performance Overview")
    st.markdown("How well does the model perform across different stores and categories?")
    
    # Selectors
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        all_stores = sorted(train.store_nbr.unique())
        default_stores = [3, 10, 20, 30, 44, 47, 50, 52, 53, 54]
        selected_stores = st.multiselect(
            "Select Stores to Compare",
            all_stores,
            default=[s for s in default_stores if s in all_stores]
        )
    with sel_col2:
        all_categories = sorted(train.family.unique())
        default_cats = ["BEVERAGES", "GROCERY I", "PRODUCE", "DAIRY", "MEATS"]
        selected_cats = st.multiselect(
            "Select Categories",
            all_categories,
            default=[c for c in default_cats if c in all_categories]
        )
    
    if not selected_stores or not selected_cats:
        st.warning("Please select at least one store and one category.")
    else:
        with st.spinner(f"Training models for {len(selected_stores)} stores × {len(selected_cats)} categories..."):
            multi_df = run_multi_store_analysis(
                train, holidays, weather, selected_stores, selected_cats)
        
        if multi_df.empty:
            st.error("No valid results. Try different stores/categories.")
        else:
            # ── Summary Metrics ──────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Avg WMAPE", f"{multi_df['xgb_wmape'].mean():.1%}")
            with m2:
                st.metric("Models Trained", f"{len(multi_df)}")
            with m3:
                win_rate = multi_df["beats_baseline"].mean() * 100
                st.metric("Beats Baseline", f"{win_rate:.0f}%")
            with m4:
                best = multi_df.loc[multi_df["xgb_wmape"].idxmin()]
                st.metric("Best Combo", f"Store {int(best['store'])}, {best['category']}")
            
            st.markdown("---")
            
            # ── Heatmap Table ────────────────────────────────
            st.subheader("📋 WMAPE by Store × Category")
            st.markdown("*Lower is better. Green = beats baseline.*")
            
            pivot = multi_df.pivot_table(
                index="store", columns="category", values="xgb_wmape")
            
            baseline_pivot = multi_df.pivot_table(
                index="store", columns="category", values="baseline_wmape")
            
            # Style the table
            def color_wmape(val):
                if pd.isna(val):
                    return ""
                if val < 0.08:
                    return "background-color: #1a5c1a; color: white"
                elif val < 0.12:
                    return "background-color: #2d8a2d; color: white"
                elif val < 0.16:
                    return "background-color: #f0ad4e; color: black"
                else:
                    return "background-color: #d9534f; color: white"
            
            styled = (pivot
                      .map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                      .style)
            
            st.dataframe(
                pivot.style
                .format("{:.1%}", na_rep="—")
                .map(color_wmape),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # ── Performance Chart ────────────────────────────
            st.subheader("📈 Model Performance Across Stores")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            for cat in selected_cats:
                cat_data = multi_df[multi_df["category"] == cat]
                if not cat_data.empty:
                    ax.plot(cat_data["store"].astype(str), 
                           cat_data["xgb_wmape"],
                           marker="o", linewidth=2, label=cat)
            
            ax.set_xlabel("Store Number")
            ax.set_ylabel("WMAPE (lower = better)")
            ax.set_title("XGBoost Performance by Store and Category")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            # ── XGBoost vs Baseline ──────────────────────────
            st.subheader("🥊 XGBoost vs Baseline — Category Averages")
            
            cat_avg = multi_df.groupby("category").agg({
                "xgb_wmape": "mean",
                "baseline_wmape": "mean",
                "beats_baseline": "mean"
            }).sort_values("xgb_wmape")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(cat_avg))
            width = 0.35
            
            bars1 = ax.bar([i - width/2 for i in x], cat_avg["baseline_wmape"], 
                          width, label="Baseline", color="orange", alpha=0.7)
            bars2 = ax.bar([i + width/2 for i in x], cat_avg["xgb_wmape"], 
                          width, label="XGBoost", color="steelblue", alpha=0.9)
            
            ax.set_ylabel("WMAPE (lower = better)")
            ax.set_title("Average WMAPE: XGBoost vs Baseline")
            ax.set_xticks(x)
            ax.set_xticklabels(cat_avg.index, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            # ── Insights ─────────────────────────────────────
            st.subheader("💡 Key Insights")
            
            worst = multi_df.loc[multi_df["xgb_wmape"].idxmax()]
            best = multi_df.loc[multi_df["xgb_wmape"].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(
                    f"**Best Performance:**\n\n"
                    f"Store {int(best['store'])} — {best['category']}\n\n"
                    f"WMAPE: {best['xgb_wmape']:.1%} "
                    f"(avg {best['avg_daily_sales']:,.0f} units/day)"
                )
            with col2:
                st.error(
                    f"**Needs Improvement:**\n\n"
                    f"Store {int(worst['store'])} — {worst['category']}\n\n"
                    f"WMAPE: {worst['xgb_wmape']:.1%} "
                    f"(avg {worst['avg_daily_sales']:,.0f} units/day)"
                )
            
            # Recommendations
            losers = multi_df[~multi_df["beats_baseline"]]
            if not losers.empty:
                st.warning(
                    f"**{len(losers)} store-category combos** where baseline beats XGBoost. "
                    f"These may benefit from: real weather data, local event integration, "
                    f"or store-specific feature engineering."
                )


# ═══════════════════════════════════════════════════════════════
# PAGE 3: ALERT PREVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "📱 Alert Preview":
    
    st.title("📱 Alert Preview")
    st.markdown("Preview what store owners would receive via WhatsApp/SMS.")
    
    # Store and category selectors
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        store_numbers = sorted(train.store_nbr.unique())
        selected_store = st.selectbox("Store", store_numbers, 
                                       index=store_numbers.index(44), key="alert_store")
    with sel_col2:
        store_data = train[train.store_nbr == selected_store]
        categories = sorted(store_data.family.unique())
        default_cat = categories.index("BEVERAGES") if "BEVERAGES" in categories else 0
        selected_category = st.selectbox("Category", categories, 
                                          index=default_cat, key="alert_cat")
    
    with st.spinner("Training model..."):
        result = train_model(store_data, holidays, weather, selected_category)
    
    if result is None:
        st.warning(f"Not enough data for this combination.")
    else:
        model = result["model"]
        X_test = result["X_test"]
        y_test = result["y_test"]
        predictions = result["predictions"]
        feat_cols = result["feat_cols"]
        
        st.markdown("---")
        
        # Show alerts for last 7 days
        st.subheader(f"📲 Last 7 Days — Store {selected_store}, {selected_category}")
        st.markdown("*Each card below is one WhatsApp/SMS message:*")
        
        for i in range(-7, 0):
            date = y_test.index[i]
            features_row = X_test.iloc[i]
            actual = y_test.values[i]
            
            pred, reasons, _ = explain_day(model, features_row, feat_cols, date)
            
            diff_pct = (pred - actual) / actual * 100 if actual > 0 else 0
            
            if diff_pct > 15:
                emoji = "🟡"
                status = "OVERSTOCKED"
                border_color = "#f0ad4e"
            elif diff_pct < -15:
                emoji = "🔴"
                status = "UNDERSTOCKED"
                border_color = "#d9534f"
            else:
                emoji = "🟢"
                status = "ON TRACK"
                border_color = "#5cb85c"
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border-left: 4px solid {border_color};
                        padding: 15px;
                        margin: 10px 0;
                        background-color: rgba(255,255,255,0.05);
                        border-radius: 0 8px 8px 0;
                    ">
                        <strong>{emoji} {date.strftime('%A, %b %d')} — {status}</strong><br><br>
                        📦 <strong>Forecast:</strong> ~{pred:,.0f} units of {selected_category}<br>
                        📊 <strong>Actual:</strong> {actual:,.0f} units<br>
                        🎯 <strong>Accuracy:</strong> {100 - abs(diff_pct):.1f}%<br><br>
                        💡 <strong>Why:</strong> {reasons}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # Alert Settings Preview
        st.subheader("⚙️ Alert Settings")
        st.markdown("*In production, store owners can customize these:*")
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.slider("Max alerts per day", 1, 10, 5, key="max_alerts")
        with s2:
            st.slider("Spike threshold (%)", 10, 50, 30, key="spike_thresh")
        with s3:
            st.selectbox("Alert channel", 
                         ["WhatsApp", "SMS", "Both"], key="channel")


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with 🔮 The Neighborhood Oracle | XGBoost + SHAP | Powered by Streamlit")