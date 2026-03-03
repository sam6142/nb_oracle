"""
Feature engineering functions.
Shared across all notebooks and scripts.
"""
import pandas as pd
import numpy as np


def build_features(df, holidays_df):
    """
    Takes raw sales data and creates all the features (clues)
    that the model uses to make predictions.
    
    Parameters:
        df: DataFrame with columns 'date', 'sales', and optionally 'onpromotion'
        holidays_df: DataFrame of holidays with columns 'date' and 'locale'
    
    Returns:
        DataFrame with all features + the 'sales' target column
    """
    data = df[["date", "sales"]].copy()
    data = data.set_index("date").sort_index()

    # Recent sales history
    data["sales_avg_7d"] = data["sales"].shift(1).rolling(7).mean()
    data["sales_avg_14d"] = data["sales"].shift(1).rolling(14).mean()
    data["sales_avg_28d"] = data["sales"].shift(1).rolling(28).mean()

    # Same day-of-week patterns
    data["sales_1w_ago"] = data["sales"].shift(7)
    data["sales_2w_ago"] = data["sales"].shift(14)
    data["sales_same_dow_avg_4w"] = (
        data["sales"].shift(7) + data["sales"].shift(14) +
        data["sales"].shift(21) + data["sales"].shift(28)
    ) / 4

    # Trend
    data["sales_trend"] = (
        data["sales"].shift(1).rolling(7).mean() -
        data["sales"].shift(8).rolling(7).mean()
    )

    # Calendar features
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["day_of_month"] = data.index.day
    data["is_weekend"] = data.index.dayofweek.isin([5, 6]).astype(int)
    data["is_payday"] = data.index.day.isin([1, 2, 15, 16]).astype(int)

    # Holidays
    national = holidays_df[holidays_df.locale == "National"]["date"].dt.date.values
    data["is_holiday"] = data.index.date
    data["is_holiday"] = data["is_holiday"].isin(national).astype(int)

    # Promotions
    if "onpromotion" in df.columns:
        promo = df[["date", "onpromotion"]].copy().set_index("date").sort_index()
        data["on_promotion"] = promo["onpromotion"].fillna(0).astype(int)

    return data


def get_feature_columns(df):
    """Returns list of feature column names (everything except 'sales')."""
    return [col for col in df.columns if col != "sales"]