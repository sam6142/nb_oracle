"""
Feature engineering functions.
Shared across all notebooks and scripts.
"""
import pandas as pd
import numpy as np


def build_features(df, holidays_df, weather_df=None, events_df=None):
    """
    Takes raw sales data and creates all the features (clues)
    that the model uses to make predictions.
    
    Parameters:
        df: DataFrame with columns 'date', 'sales', and optionally 'onpromotion'
        holidays_df: DataFrame of holidays with columns 'date' and 'locale'
        weather_df: (optional) DataFrame with columns 'date', 'temp_high', etc.
                    If None, weather features are skipped.
        events_df: (optional) DataFrame from events_to_features() with columns
                   'date', 'nearby_events', 'city_events', 'event_score', etc.
                   If None, event features are skipped.
    
    Returns:
        DataFrame with all features + the 'sales' target column
    """
    data = df[["date", "sales"]].copy()
    data = data.set_index("date").sort_index()

    # ── Recent sales history ─────────────────────────────────────
    data["sales_avg_7d"] = data["sales"].shift(1).rolling(7).mean()
    data["sales_avg_14d"] = data["sales"].shift(1).rolling(14).mean()
    data["sales_avg_28d"] = data["sales"].shift(1).rolling(28).mean()

    # ── Same day-of-week patterns ────────────────────────────────
    data["sales_1w_ago"] = data["sales"].shift(7)
    data["sales_2w_ago"] = data["sales"].shift(14)
    data["sales_same_dow_avg_4w"] = (
        data["sales"].shift(7) + data["sales"].shift(14) +
        data["sales"].shift(21) + data["sales"].shift(28)
    ) / 4

    # ── Trend ────────────────────────────────────────────────────
    data["sales_trend"] = (
        data["sales"].shift(1).rolling(7).mean() -
        data["sales"].shift(8).rolling(7).mean()
    )

    # ── Calendar features ────────────────────────────────────────
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_month"] = data.index.day
    data["is_weekend"] = data.index.dayofweek.isin([5, 6]).astype(int)
    data["is_payday"] = data.index.day.isin([1, 2, 15, 16]).astype(int)

    # ── Holidays ─────────────────────────────────────────────────
    national = holidays_df[holidays_df.locale == "National"]["date"].dt.date.values
    data["is_holiday"] = data.index.date
    data["is_holiday"] = data["is_holiday"].isin(national).astype(int)

    # ── Promotions ───────────────────────────────────────────────
    if "onpromotion" in df.columns:
        promo = df[["date", "onpromotion"]].copy().set_index("date").sort_index()
        data["on_promotion"] = promo["onpromotion"].fillna(0).astype(int)

    # ── Weather features (if provided) ───────────────────────────
    if weather_df is not None:
        data = _add_weather_features(data, weather_df)

    # ── Event features (if provided) ─────────────────────────────
    if events_df is not None:
        data = _add_event_features(data, events_df)

    return data


def _add_weather_features(data, weather_df):
    """
    Adds weather-based features to the dataset.
    
    Key insight: it's not the ABSOLUTE temperature that drives sales,
    it's the CHANGE. A 90°F day after a week of 90°F days is normal.
    A 90°F day after a week of 70°F days triggers different buying.
    """
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.set_index("date").sort_index()
    
    # Basic weather values
    data["temp_high"] = weather["temp_high"]
    data["temp_feels_like"] = weather["temp_feels_like"]
    data["is_precipitation"] = weather["is_precipitation"]
    data["precipitation_mm"] = weather["precipitation_mm"]
    
    # DELTA features — these are more predictive than raw values
    data["temp_delta_vs_yesterday"] = weather["temp_high"].diff(1)
    
    weekly_avg_temp = weather["temp_high"].rolling(7).mean()
    data["temp_delta_vs_weekly_avg"] = weather["temp_high"] - weekly_avg_temp
    
    # Weather severity: 0=normal, 1=notable, 2=extreme
    data["weather_severity"] = 0
    data.loc[data["temp_delta_vs_yesterday"].abs() > 5, "weather_severity"] = 1
    data.loc[
        (data["temp_delta_vs_yesterday"].abs() > 10) |
        (data["precipitation_mm"] > 20),
        "weather_severity"
    ] = 2
    
    # Is it a hot day? (above 75th percentile of recent temps)
    temp_75th = weather["temp_high"].rolling(28, min_periods=7).quantile(0.75)
    data["is_hot_day"] = (weather["temp_high"] > temp_75th).astype(int)
    
    # Is it a cold day? (below 25th percentile of recent temps)
    temp_25th = weather["temp_high"].rolling(28, min_periods=7).quantile(0.25)
    data["is_cold_day"] = (weather["temp_high"] < temp_25th).astype(int)
    
    # Rain streak: how many consecutive days has it been raining?
    rain_streak = weather["is_precipitation"].copy()
    streak = 0
    streaks = []
    for val in rain_streak:
        if val == 1:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    data["rain_streak_days"] = streaks
    
    return data


def _add_event_features(data, events_df):
    """
    Adds event-based features to the dataset.
    
    Key insight: nearby events BOOST demand, city-level events may
    REDUCE it by drawing customers elsewhere. The event_score captures
    this net effect.
    
    nearby = within 2 miles (block parties, local festivals → more foot traffic)
    city = 2-10 miles (Fenway, TD Garden → may pull customers away)
    """
    events = events_df.copy()
    events["date"] = pd.to_datetime(events["date"])
    events = events.set_index("date")
    
    event_cols = ["nearby_events", "nearby_attendance", "city_events",
                  "city_attendance", "has_sports_nearby", "has_sports_city",
                  "has_music", "is_marathon", "event_score"]
    
    for col in event_cols:
        if col in events.columns:
            data[col] = events[col]
    
    # Fill missing days with 0 (no events)
    for col in event_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0).astype(float)
    
    return data

def validate_features(df, expected_columns):
    """
    Check that a feature DataFrame matches what the model expects.
    Raises an error if columns are missing or extra.
    
    Parameters:
        df: DataFrame of features
        expected_columns: list of column names the model was trained with
    
    Returns:
        DataFrame with columns in the correct order
    """
    df_cols = set(get_feature_columns(df))
    expected = set(expected_columns)
    
    missing = expected - df_cols
    extra = df_cols - expected
    
    if missing:
        raise ValueError(
            f"Feature mismatch: model expects {len(missing)} features that are missing: "
            f"{sorted(missing)}"
        )
    
    if extra:
        # Extra columns are okay — just ignore them with a warning
        import logging
        logging.getLogger(__name__).warning(
            f"Ignoring {len(extra)} extra features not used by model: {sorted(extra)}"
        )
    
    # Return columns in the exact order the model expects
    return df[expected_columns]


def build_prediction_features(recent_sales_df, holidays_df, weather_hist_df,
                                forecast_weather_row, events_hist_df=None,
                                forecast_event_row=None):
    """
    Build features for a single future date prediction.
    
    THIS IS THE ONLY WAY to build features for live predictions.
    It uses the exact same build_features() function as training,
    ensuring no training/serving skew.
    
    Parameters:
        recent_sales_df: last 60+ days of sales (columns: date, sales, onpromotion)
        holidays_df: holidays DataFrame
        weather_hist_df: historical weather DataFrame
        forecast_weather_row: dict or Series with forecast day's weather
        events_hist_df: (optional) historical event features DataFrame
        forecast_event_row: (optional) dict with forecast day's event features
    
    Returns:
        Series of features for the forecast date (single row)
    """
    import pandas as pd
    
    forecast_date = pd.Timestamp(forecast_weather_row["date"])
    
    # Create the future row with unknown sales
    future_row = pd.DataFrame({
        "date": [forecast_date],
        "sales": [0],
        "onpromotion": [0],
    })
    
    # Combine recent history + future date
    combined_sales = pd.concat([
        recent_sales_df[["date", "sales", "onpromotion"]].tail(60),
        future_row
    ]).reset_index(drop=True)
    
    # Build weather: filter history to matching dates + add forecast
    combined_dates = pd.to_datetime(combined_sales["date"])
    weather_filtered = weather_hist_df[
        weather_hist_df["date"].isin(combined_dates)
    ].copy()
    
    forecast_weather = pd.DataFrame([{
        "date": forecast_date,
        "temp_high": forecast_weather_row["temp_high"],
        "temp_feels_like": forecast_weather_row["temp_feels_like"],
        "precipitation_mm": forecast_weather_row["precipitation_mm"],
        "is_precipitation": forecast_weather_row["is_precipitation"],
        "humidity": forecast_weather_row.get("humidity", 65),
    }])
    
    weather_combined = pd.concat(
        [weather_filtered, forecast_weather]
    ).drop_duplicates(subset="date")
    
    # Build events if provided
    events_combined = None
    if events_hist_df is not None:
        events_filtered = events_hist_df[
            events_hist_df["date"].isin(combined_dates)
        ].copy()
        
        if forecast_event_row is not None:
            forecast_event = pd.DataFrame([forecast_event_row])
            forecast_event["date"] = forecast_date
            events_combined = pd.concat(
                [events_filtered, forecast_event]
            ).reset_index(drop=True)
            events_combined["date"] = pd.to_datetime(events_combined["date"])
        else:
            events_combined = events_filtered
    
    # Use the SAME build_features() as training
    features = build_features(
        combined_sales, holidays_df,
        weather_df=weather_combined,
        events_df=events_combined
    ).dropna()
    
    if len(features) == 0:
        return None
    
    # Return the last row (the forecast date)
    return features.iloc[-1]

def get_feature_columns(df):
    """Returns list of feature column names (everything except 'sales')."""
    return [col for col in df.columns if col != "sales"]