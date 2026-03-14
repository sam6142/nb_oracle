"""
Data quality validation for all incoming data sources.

Checks 6 dimensions for each data source before training or inference:
  1. ACCURACY     — values are within plausible physical/business ranges
  2. COMPLETENESS — no missing values or unexplained date gaps
  3. CONSISTENCY  — category names and date formats are uniform
  4. TIMELINESS   — data is recent enough to produce reliable predictions
  5. UNIQUENESS   — no duplicate rows that would inflate training signal
  6. VALIDITY     — correct data types (dates are dates, sales are numbers)

Usage:
    from data.validation import validate_sales_data, validate_weather_data, validate_events_data

    report = validate_sales_data(df)
    if not report["passed"]:
        print(report["errors"])   # stop — data can't be used
    print(report["warnings"])     # may continue, but quality is degraded
"""
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    VALIDATION_MAX_DAILY_SALES,
    VALIDATION_MAX_TEMP_F,
    VALIDATION_MIN_TEMP_F,
    VALIDATION_MAX_PRECIP_MM,
    VALIDATION_STALE_SALES_DAYS,
    VALIDATION_STALE_WEATHER_DAYS,
)


# ══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def _make_report(passed, warnings, errors, stats):
    """
    Build the standard validation result dict.

    Returns:
        dict with keys: passed (bool), warnings (list[str]),
                        errors (list[str]), stats (dict)
    """
    return {"passed": passed, "warnings": warnings, "errors": errors, "stats": stats}


def _is_parseable_as_dates(series):
    """Return True if the series can be coerced to datetime without error."""
    try:
        pd.to_datetime(series)
        return True
    except Exception:
        return False


def _latest_date_staleness(date_series, stale_threshold_days):
    """
    Return (latest_date, days_stale) for a date column.
    Returns (None, None) if the column can't be parsed.
    """
    try:
        dates = pd.to_datetime(date_series, errors="coerce")
        latest = dates.max()
        if pd.isna(latest):
            return None, None
        days_stale = (datetime.now() - latest).days
        return latest, days_stale
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════════════════
# SALES DATA VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_sales_data(df):
    """
    Validate sales data across 6 quality dimensions.

    Designed for the per-category DataFrames used in model/train.py,
    but also works on the full multi-category train.csv.

    Parameters:
        df: DataFrame with columns: date, sales, and optionally family, store_nbr

    Returns:
        dict with: passed (bool), warnings (list), errors (list), stats (dict)
        passed=False means the data cannot safely be used for training.
    """
    errors, warnings = [], []

    _sales_validity(df, errors)
    _sales_accuracy(df, errors, warnings)
    _sales_completeness(df, errors, warnings)
    _sales_consistency(df, warnings)
    _sales_timeliness(df, warnings)
    _sales_uniqueness(df, errors)

    return _make_report(passed=len(errors) == 0, warnings=warnings,
                        errors=errors, stats=_sales_stats(df))


def _sales_validity(df, errors):
    """VALIDITY: required columns exist and have usable types."""
    if "date" not in df.columns:
        errors.append("VALIDITY: Missing required column 'date'")
    elif not pd.api.types.is_datetime64_any_dtype(df["date"]):
        if not _is_parseable_as_dates(df["date"]):
            errors.append("VALIDITY: 'date' column cannot be parsed as dates")

    if "sales" not in df.columns:
        errors.append("VALIDITY: Missing required column 'sales'")
    elif not pd.api.types.is_numeric_dtype(df["sales"]):
        errors.append("VALIDITY: 'sales' column must be numeric — found non-numeric values")


def _sales_accuracy(df, errors, warnings):
    """ACCURACY: sales values are within realistic bodega ranges."""
    if "sales" not in df.columns:
        return

    negative_count = (df["sales"] < 0).sum()
    if negative_count > 0:
        errors.append(
            f"ACCURACY: {negative_count} rows have negative sales — "
            "sales cannot be negative (returns should be zeroed out)"
        )

    high_count = (df["sales"] > VALIDATION_MAX_DAILY_SALES).sum()
    if high_count > 0:
        warnings.append(
            f"ACCURACY: {high_count} rows exceed {VALIDATION_MAX_DAILY_SALES:,} units/day "
            f"(max seen: {df['sales'].max():.0f}) — verify this is realistic for your store"
        )


def _sales_completeness(df, errors, warnings):
    """COMPLETENESS: no null values, no gaps longer than 7 days."""
    if "sales" in df.columns:
        null_count = df["sales"].isnull().sum()
        if null_count > 0:
            errors.append(f"COMPLETENESS: {null_count} rows have null sales values")

    if "date" not in df.columns:
        return

    null_dates = df["date"].isnull().sum()
    if null_dates > 0:
        errors.append(f"COMPLETENESS: {null_dates} rows have null dates")
        return

    # Check for gaps: lag features break when the time series has holes > 7 days
    if pd.api.types.is_datetime64_any_dtype(df["date"]) or _is_parseable_as_dates(df["date"]):
        dates = pd.to_datetime(df["date"]).sort_values().drop_duplicates()
        gaps = dates.diff().dt.days.dropna()
        long_gaps = gaps[gaps > 7]
        if len(long_gaps) > 0:
            warnings.append(
                f"COMPLETENESS: {len(long_gaps)} gap(s) longer than 7 days detected "
                f"(longest: {long_gaps.max():.0f} days) — "
                "lag and rolling features will contain NaN around these gaps"
            )


def _sales_consistency(df, warnings):
    """CONSISTENCY: category names are uniform (catches 'Beverage' vs 'BEVERAGES')."""
    if "family" not in df.columns:
        return

    # Group variants by their normalized (uppercased, stripped) form
    normalized = {}
    for name in df["family"].dropna().unique():
        key = name.strip().upper()
        normalized.setdefault(key, []).append(name)

    for norm_name, variants in normalized.items():
        if len(variants) > 1:
            warnings.append(
                f"CONSISTENCY: Category '{norm_name}' appears in multiple forms: {variants} — "
                "this can cause category filtering to silently miss rows"
            )


def _sales_timeliness(df, warnings):
    """TIMELINESS: the most recent sales row isn't stale."""
    if "date" not in df.columns:
        return

    latest, days_stale = _latest_date_staleness(df["date"], VALIDATION_STALE_SALES_DAYS)
    if latest is None:
        warnings.append("TIMELINESS: Could not determine latest sales date")
    elif days_stale > VALIDATION_STALE_SALES_DAYS:
        warnings.append(
            f"TIMELINESS: Latest sales date is {latest.date()} "
            f"({days_stale} days ago) — predictions may be based on stale data"
        )


def _sales_uniqueness(df, errors):
    """UNIQUENESS: no duplicate (store_nbr, family, date) combinations."""
    key_cols = [c for c in ["store_nbr", "family", "date"] if c in df.columns]
    if not key_cols:
        return

    dupe_count = df.duplicated(subset=key_cols).sum()
    if dupe_count > 0:
        errors.append(
            f"UNIQUENESS: {dupe_count} duplicate rows for the same "
            f"({', '.join(key_cols)}) combination — "
            "duplicates inflate training signal and must be removed"
        )


def _sales_stats(df):
    """Compute a concise summary for the validation report."""
    stats = {"n_rows": len(df)}
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        stats["date_min"] = str(dates.min().date()) if not dates.isna().all() else None
        stats["date_max"] = str(dates.max().date()) if not dates.isna().all() else None
        stats["n_days"] = int(dates.nunique())
    if "sales" in df.columns:
        stats["sales_mean"] = round(float(df["sales"].mean()), 2)
        stats["sales_max"] = float(df["sales"].max())
        stats["null_sales"] = int(df["sales"].isnull().sum())
    if "family" in df.columns:
        stats["n_categories"] = int(df["family"].nunique())
    return stats


# ══════════════════════════════════════════════════════════════════
# WEATHER DATA VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_weather_data(df):
    """
    Validate weather data across 6 quality dimensions.

    Parameters:
        df: DataFrame with columns: date, temp_high, temp_feels_like,
            precipitation_mm, is_precipitation, humidity

    Returns:
        dict with: passed (bool), warnings (list), errors (list), stats (dict)
    """
    errors, warnings = [], []

    _weather_validity(df, errors)
    _weather_accuracy(df, errors, warnings)
    _weather_completeness(df, errors, warnings)
    _weather_consistency(df, warnings)
    _weather_timeliness(df, warnings)
    _weather_uniqueness(df, errors)

    return _make_report(passed=len(errors) == 0, warnings=warnings,
                        errors=errors, stats=_weather_stats(df))


def _weather_validity(df, errors):
    """VALIDITY: required columns exist with numeric types."""
    required = ["date", "temp_high", "precipitation_mm"]
    for col in required:
        if col not in df.columns:
            errors.append(f"VALIDITY: Missing required weather column '{col}'")

    numeric_cols = ["temp_high", "temp_feels_like", "precipitation_mm", "is_precipitation"]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"VALIDITY: Weather column '{col}' must be numeric")


def _weather_accuracy(df, errors, warnings):
    """ACCURACY: temperatures and precipitation are physically possible."""
    if "temp_high" in df.columns:
        too_hot = (df["temp_high"] > VALIDATION_MAX_TEMP_F).sum()
        too_cold = (df["temp_high"] < VALIDATION_MIN_TEMP_F).sum()
        if too_hot > 0:
            errors.append(
                f"ACCURACY: {too_hot} rows have temp_high > {VALIDATION_MAX_TEMP_F}°F — "
                "likely a unit error (°C values in a °F column?)"
            )
        if too_cold > 0:
            errors.append(
                f"ACCURACY: {too_cold} rows have temp_high < {VALIDATION_MIN_TEMP_F}°F — "
                "likely a data error"
            )

    if "precipitation_mm" in df.columns:
        neg_precip = (df["precipitation_mm"] < 0).sum()
        extreme_precip = (df["precipitation_mm"] > VALIDATION_MAX_PRECIP_MM).sum()
        if neg_precip > 0:
            errors.append(f"ACCURACY: {neg_precip} rows have negative precipitation_mm")
        if extreme_precip > 0:
            warnings.append(
                f"ACCURACY: {extreme_precip} rows have precipitation_mm > {VALIDATION_MAX_PRECIP_MM} "
                "— verify your weather data source"
            )


def _weather_completeness(df, errors, warnings):
    """COMPLETENESS: no nulls in critical columns; optional columns are warned."""
    for col in ["date", "temp_high", "precipitation_mm"]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(
                    f"COMPLETENESS: {null_count} null values in required weather column '{col}'"
                )

    for col in ["temp_feels_like", "humidity", "is_precipitation"]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                warnings.append(
                    f"COMPLETENESS: {null_count} null values in optional weather column "
                    f"'{col}' (will be filled with 0 during feature engineering)"
                )


def _weather_consistency(df, warnings):
    """CONSISTENCY: date format is uniform across all rows."""
    if "date" not in df.columns:
        return
    # Try to detect mixed formats: if coercion with a strict format fails on some rows
    # but to_datetime succeeds, pandas guessed mixed formats.
    try:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        null_after = int(parsed.isnull().sum())
        original_null = int(df["date"].isnull().sum())
        if null_after > original_null:
            warnings.append(
                f"CONSISTENCY: {null_after - original_null} weather dates could not be parsed "
                "— mixed date formats may be present"
            )
    except Exception:
        pass


def _weather_timeliness(df, warnings):
    """TIMELINESS: weather data is not stale."""
    if "date" not in df.columns:
        return
    latest, days_stale = _latest_date_staleness(df["date"], VALIDATION_STALE_WEATHER_DAYS)
    if latest is None:
        warnings.append("TIMELINESS: Could not determine latest weather date")
    elif days_stale > VALIDATION_STALE_WEATHER_DAYS:
        warnings.append(
            f"TIMELINESS: Latest weather date is {latest.date()} "
            f"({days_stale} days ago) — weather features may be outdated"
        )


def _weather_uniqueness(df, errors):
    """UNIQUENESS: no duplicate dates (one row per day)."""
    if "date" not in df.columns:
        return
    dupe_count = df.duplicated(subset=["date"]).sum()
    if dupe_count > 0:
        errors.append(
            f"UNIQUENESS: {dupe_count} duplicate dates in weather data — "
            "keep only one row per date before training"
        )


def _weather_stats(df):
    """Compute a concise summary for the weather validation report."""
    stats = {"n_rows": len(df)}
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        stats["date_min"] = str(dates.min().date()) if not dates.isna().all() else None
        stats["date_max"] = str(dates.max().date()) if not dates.isna().all() else None
    if "temp_high" in df.columns:
        stats["temp_high_mean"] = round(float(df["temp_high"].mean()), 1)
        stats["temp_high_min"] = float(df["temp_high"].min())
        stats["temp_high_max"] = float(df["temp_high"].max())
    if "precipitation_mm" in df.columns:
        stats["rainy_days"] = int((df["precipitation_mm"] > 0).sum())
    return stats


# ══════════════════════════════════════════════════════════════════
# EVENTS DATA VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_events_data(df):
    """
    Validate events feature data across 6 quality dimensions.

    Parameters:
        df: DataFrame from events_to_features() with columns:
            date, nearby_events, nearby_attendance, city_events,
            city_attendance, has_sports_nearby, has_sports_city,
            has_music, is_marathon, event_score

    Returns:
        dict with: passed (bool), warnings (list), errors (list), stats (dict)
    """
    errors, warnings = [], []

    _events_validity(df, errors)
    _events_accuracy(df, errors, warnings)
    _events_completeness(df, errors, warnings)
    _events_consistency(df, warnings)
    _events_timeliness(df, warnings)
    _events_uniqueness(df, errors)

    return _make_report(passed=len(errors) == 0, warnings=warnings,
                        errors=errors, stats=_events_stats(df))


def _events_validity(df, errors):
    """VALIDITY: required column exists and numeric columns are numeric."""
    if "date" not in df.columns:
        errors.append("VALIDITY: Missing required column 'date' in events data")

    numeric_cols = ["nearby_events", "city_events", "event_score",
                    "nearby_attendance", "city_attendance"]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"VALIDITY: Events column '{col}' must be numeric")


def _events_accuracy(df, errors, warnings):
    """ACCURACY: counts are non-negative, attendance is within reason."""
    for col in ["nearby_events", "city_events"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(
                    f"ACCURACY: {neg_count} negative values in '{col}' — "
                    "event counts cannot be negative"
                )

    # 1 million attendees/day in a single column is extreme
    for col in ["nearby_attendance", "city_attendance"]:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"ACCURACY: {neg_count} negative values in '{col}'")
            extreme_count = (df[col] > 1_000_000).sum()
            if extreme_count > 0:
                warnings.append(
                    f"ACCURACY: {extreme_count} rows have attendance > 1,000,000 in '{col}' "
                    "— verify data source"
                )


def _events_completeness(df, errors, warnings):
    """COMPLETENESS: no null dates; null event columns can be filled with 0."""
    if "date" in df.columns:
        null_dates = df["date"].isnull().sum()
        if null_dates > 0:
            errors.append(f"COMPLETENESS: {null_dates} null dates in events data")

    for col in ["nearby_events", "city_events", "event_score"]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                warnings.append(
                    f"COMPLETENESS: {null_count} null values in events column '{col}' "
                    "(will be filled with 0 — treated as no-event day)"
                )


def _events_consistency(df, warnings):
    """CONSISTENCY: date format is uniform across all rows."""
    if "date" not in df.columns:
        return
    try:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        unparseable = parsed.isnull().sum() - df["date"].isnull().sum()
        if unparseable > 0:
            warnings.append(
                f"CONSISTENCY: {unparseable} event dates could not be parsed "
                "— mixed date formats may be present"
            )
    except Exception:
        pass


def _events_timeliness(df, warnings):
    """TIMELINESS: events data covers reasonably recent dates."""
    if "date" not in df.columns:
        return
    # Events are inherently forward-looking; warn only if significantly stale
    latest, days_stale = _latest_date_staleness(df["date"], stale_threshold_days=7)
    if latest is None:
        warnings.append("TIMELINESS: Could not determine latest events date")
    elif days_stale > 7:
        warnings.append(
            f"TIMELINESS: Latest events date is {latest.date()} "
            f"({days_stale} days ago) — upcoming events may not be captured"
        )


def _events_uniqueness(df, errors):
    """UNIQUENESS: no duplicate dates in the events feature table."""
    if "date" not in df.columns:
        return
    dupe_count = df.duplicated(subset=["date"]).sum()
    if dupe_count > 0:
        errors.append(
            f"UNIQUENESS: {dupe_count} duplicate dates in events data — "
            "keep only one row per date"
        )


def _events_stats(df):
    """Compute a concise summary for the events validation report."""
    stats = {"n_rows": len(df)}
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        stats["date_min"] = str(dates.min().date()) if not dates.isna().all() else None
        stats["date_max"] = str(dates.max().date()) if not dates.isna().all() else None
    if "nearby_events" in df.columns:
        stats["days_with_nearby_events"] = int((df["nearby_events"] > 0).sum())
    if "city_events" in df.columns:
        stats["days_with_city_events"] = int((df["city_events"] > 0).sum())
    return stats
