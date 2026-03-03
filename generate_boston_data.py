"""
Generate realistic mock sales data for a Boston bodega.

Output format is IDENTICAL to the Favorita dataset so all existing
notebooks and pipelines work with zero changes. Just point DATA_RAW
to the boston data folder instead.

Usage: python generate_boston_data.py
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta

OUTPUT_DIR = Path(r"C:\Users\syeds\OneDrive\Desktop\nboracle\nb_oracle\data\raw\boston-bodega")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# STEP 1: Define the store
# ══════════════════════════════════════════════════════════════

STORE = {
    "store_nbr": 1,
    "city": "Dorchester",
    "state": "Massachusetts",
    "type": "A",
    "cluster": 1,
}

# Same 33 categories as Favorita, with Boston-appropriate daily sales ranges
# (base_mean, base_std) — average units sold per day and variability
CATEGORIES = {
    "AUTOMOTIVE":              (3, 2),
    "BABY CARE":               (5, 3),
    "BEAUTY":                  (8, 4),
    "BEVERAGES":               (120, 35),
    "BOOKS":                   (2, 2),
    "BREAD/BAKERY":            (65, 20),
    "CELEBRATION":             (4, 3),
    "CLEANING":                (25, 10),
    "DAIRY":                   (80, 25),
    "DELI":                    (30, 12),
    "EGGS":                    (40, 15),
    "FROZEN FOODS":            (20, 8),
    "GROCERY I":               (150, 40),
    "GROCERY II":              (15, 7),
    "HARDWARE":                (3, 2),
    "HOME AND KITCHEN I":      (8, 5),
    "HOME AND KITCHEN II":     (5, 3),
    "HOME APPLIANCES":         (2, 2),
    "HOME CARE":               (12, 6),
    "LADIESWEAR":              (4, 3),
    "LAWN AND GARDEN":         (3, 2),
    "LINGERIE":                (3, 2),
    "LIQUOR,WINE,BEER":        (35, 15),
    "MAGAZINES":               (2, 2),
    "MEATS":                   (45, 18),
    "PERSONAL CARE":           (18, 8),
    "PET SUPPLIES":            (6, 4),
    "PLAYERS AND ELECTRONICS": (3, 2),
    "POULTRY":                 (35, 14),
    "PREPARED FOODS":          (25, 10),
    "PRODUCE":                 (90, 30),
    "SCHOOL AND OFFICE SUPPLIES": (4, 3),
    "SEAFOOD":                 (15, 8),
}

# ══════════════════════════════════════════════════════════════
# STEP 2: Fetch real historical weather for Boston
# ══════════════════════════════════════════════════════════════

def fetch_boston_weather(start_date, end_date):
    """Fetch real historical weather for Boston from Open-Meteo."""
    print("Fetching real Boston weather from Open-Meteo...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 42.3601,
        "longitude": -71.0589,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_sum,rain_sum,snowfall_sum",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print("Falling back to simulated weather")
        return None
    
    data = response.json()
    
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_high": data["daily"]["temperature_2m_max"],
        "temp_low": data["daily"]["temperature_2m_min"],
        "temp_feels_like": data["daily"]["apparent_temperature_max"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
        "rain_mm": data["daily"]["rain_sum"],
        "snow_mm": data["daily"]["snowfall_sum"],
    })
    
    # Fill any nulls
    df = df.fillna(0)
    
    print(f"  Got {len(df)} days of real Boston weather ✅")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 3: Generate sales data influenced by real patterns
# ══════════════════════════════════════════════════════════════

def generate_sales(dates, weather_df):
    """
    Generate realistic daily sales for all categories.
    
    Sales are influenced by:
    - Day of week (weekends higher)
    - Month/season (summer higher for beverages, winter for soups)
    - Weather (hot days = more beverages, cold = more hot food)
    - Holidays (spikes before, dips during)
    - Payday effect (1st and 15th)
    - Random noise
    - Gradual trend (slight growth over time)
    """
    records = []
    row_id = 0
    
    # Pre-compute weather lookup
    weather_lookup = {}
    if weather_df is not None:
        for _, row in weather_df.iterrows():
            weather_lookup[row["date"].date()] = row
    
    for date in dates:
        dt = pd.Timestamp(date)
        dow = dt.dayofweek          # 0=Mon, 6=Sun
        month = dt.month
        day = dt.day
        day_of_year = dt.dayofyear
        
        # Get weather for this date
        w = weather_lookup.get(dt.date(), None)
        temp = w["temp_high"] if w is not None else 65
        precip = w["precipitation_mm"] if w is not None else 0
        snow = w.get("snow_mm", 0) if w is not None else 0
        
        # ── Global multipliers (affect all categories) ───────
        
        # Day of week effect
        dow_multiplier = {
            0: 0.85,  # Monday - slow
            1: 0.88,  # Tuesday
            2: 0.90,  # Wednesday
            3: 0.92,  # Thursday
            4: 1.05,  # Friday - pre-weekend shopping
            5: 1.20,  # Saturday - busiest
            6: 1.10,  # Sunday
        }[dow]
        
        # Payday effect (1st, 2nd, 15th, 16th)
        payday_mult = 1.15 if day in [1, 2, 15, 16] else 1.0
        
        # Severe weather reduces foot traffic
        if snow > 10:
            traffic_mult = 0.5   # Blizzard - big drop
        elif snow > 2:
            traffic_mult = 0.75  # Snow - moderate drop
        elif precip > 15:
            traffic_mult = 0.8   # Heavy rain
        elif precip > 5:
            traffic_mult = 0.9   # Light rain
        else:
            traffic_mult = 1.0
        
        # Gradual growth trend (2% per year)
        days_from_start = (dt - dates[0]).days
        trend_mult = 1 + (days_from_start / 365) * 0.02
        
        for family, (base_mean, base_std) in CATEGORIES.items():
            
            # ── Category-specific weather effects ────────────
            weather_mult = 1.0
            
            if family in ["BEVERAGES", "FROZEN FOODS"]:
                # Hot weather boosts cold items
                if temp > 85:
                    weather_mult = 1.35
                elif temp > 75:
                    weather_mult = 1.15
                elif temp < 35:
                    weather_mult = 0.85
            
            elif family in ["PREPARED FOODS", "BREAD/BAKERY"]:
                # Cold weather boosts comfort food
                if temp < 35:
                    weather_mult = 1.25
                elif temp < 50:
                    weather_mult = 1.10
                elif temp > 85:
                    weather_mult = 0.90
            
            elif family in ["LIQUOR,WINE,BEER"]:
                # Warm weekends boost alcohol
                if temp > 70 and dow >= 4:
                    weather_mult = 1.30
                elif temp > 70:
                    weather_mult = 1.10
            
            elif family in ["PRODUCE", "DAIRY", "EGGS"]:
                # Stable essentials, slight seasonal variation
                seasonal = 1 + 0.05 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
                weather_mult = seasonal
            
            elif family in ["CLEANING", "HOME CARE"]:
                # Spring cleaning boost
                if month in [3, 4, 5]:
                    weather_mult = 1.15
            
            elif family in ["SCHOOL AND OFFICE SUPPLIES"]:
                # Back to school spike
                if month in [8, 9]:
                    weather_mult = 2.5
                elif month in [1]:
                    weather_mult = 1.5
            
            elif family in ["CELEBRATION"]:
                # Holiday spikes
                if month == 12:
                    weather_mult = 3.0
                elif month in [2, 10]:  # Valentine's, Halloween
                    weather_mult = 2.0
            
            # ── Seasonal baseline shift ──────────────────────
            # Some categories are naturally more seasonal
            if family in ["LAWN AND GARDEN"]:
                if month in [4, 5, 6, 7, 8]:
                    seasonal_mult = 2.0
                else:
                    seasonal_mult = 0.3
            elif family in ["SEAFOOD"]:
                # Lent / fish Fridays
                if month in [2, 3, 4] and dow == 4:
                    seasonal_mult = 1.8
                else:
                    seasonal_mult = 1.0
            else:
                seasonal_mult = 1.0
            
            # ── Promotion (random, ~15% of days) ─────────────
            on_promo = 1 if np.random.random() < 0.15 else 0
            promo_mult = 1.20 if on_promo else 1.0
            
            # ── Calculate final sales ────────────────────────
            total_mult = (dow_multiplier * payday_mult * traffic_mult * 
                         trend_mult * weather_mult * seasonal_mult * promo_mult)
            
            raw_sales = base_mean * total_mult + np.random.normal(0, base_std)
            
            # Sales can't be negative
            sales = max(0, round(raw_sales))
            
            records.append({
                "id": row_id,
                "date": dt.strftime("%Y-%m-%d"),
                "store_nbr": STORE["store_nbr"],
                "family": family,
                "sales": sales,
                "onpromotion": on_promo,
            })
            row_id += 1
    
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# STEP 4: Generate US holidays
# ══════════════════════════════════════════════════════════════

def generate_holidays(start_year, end_year):
    """Generate US holidays in the same format as Favorita."""
    holidays = []
    
    fixed_holidays = [
        ("01-01", "New Year's Day"),
        ("07-04", "Independence Day"),
        ("12-25", "Christmas"),
        ("12-31", "New Year's Eve"),
    ]
    
    # Variable holidays (approximate dates)
    variable_holidays = {
        2024: [
            ("01-15", "MLK Day"),
            ("02-14", "Valentine's Day"),
            ("02-19", "Presidents Day"),
            ("03-31", "Easter"),
            ("05-27", "Memorial Day"),
            ("09-02", "Labor Day"),
            ("10-31", "Halloween"),
            ("11-28", "Thanksgiving"),
            ("11-29", "Black Friday"),
        ],
        2025: [
            ("01-20", "MLK Day"),
            ("02-14", "Valentine's Day"),
            ("02-17", "Presidents Day"),
            ("04-20", "Easter"),
            ("05-26", "Memorial Day"),
            ("09-01", "Labor Day"),
            ("10-31", "Halloween"),
            ("11-27", "Thanksgiving"),
            ("11-28", "Black Friday"),
        ],
        2026: [
            ("01-19", "MLK Day"),
            ("02-14", "Valentine's Day"),
            ("02-16", "Presidents Day"),
            ("04-05", "Easter"),
            ("05-25", "Memorial Day"),
            ("09-07", "Labor Day"),
            ("10-31", "Halloween"),
            ("11-26", "Thanksgiving"),
            ("11-27", "Black Friday"),
        ],
    }
    
    for year in range(start_year, end_year + 1):
        for md, name in fixed_holidays:
            holidays.append({
                "date": f"{year}-{md}",
                "type": "Holiday",
                "locale": "National",
                "locale_name": "United States",
                "description": name,
                "transferred": False,
            })
        
        if year in variable_holidays:
            for md, name in variable_holidays[year]:
                holidays.append({
                    "date": f"{year}-{md}",
                    "type": "Holiday",
                    "locale": "National",
                    "locale_name": "United States",
                    "description": name,
                    "transferred": False,
                })
    
    return pd.DataFrame(holidays)


# ══════════════════════════════════════════════════════════════
# STEP 5: Generate everything
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Boston Bodega Dataset")
    print("=" * 60)
    
    # Date range: Jan 2024 to yesterday
    start_date = "2024-01-01"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = yesterday
    
    dates = pd.date_range(start_date, end_date)
    print(f"\nDate range: {start_date} to {end_date} ({len(dates)} days)")
    
    # Fetch real Boston weather
    weather = fetch_boston_weather(start_date, end_date)
    
    # Save weather separately
    if weather is not None:
        weather.to_csv(OUTPUT_DIR / "weather_boston.csv", index=False)
        print(f"Saved weather: {OUTPUT_DIR / 'weather_boston.csv'}")
    
    # Generate sales
    print(f"\nGenerating sales for {len(CATEGORIES)} categories...")
    train_df = generate_sales(dates, weather)
    print(f"  Generated {len(train_df):,} sales records")
    
    # Generate stores.csv
    stores_df = pd.DataFrame([STORE])
    
    # Generate holidays
    holidays_df = generate_holidays(2024, 2026)
    
    # Save everything in Favorita format
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    stores_df.to_csv(OUTPUT_DIR / "stores.csv", index=False)
    holidays_df.to_csv(OUTPUT_DIR / "holidays_events.csv", index=False)
    
    # Also save weather with the right columns for our pipeline
    if weather is not None:
        weather_pipeline = weather[["date", "temp_high", "temp_feels_like", "precipitation_mm"]].copy()
        weather_pipeline["is_precipitation"] = (weather_pipeline["precipitation_mm"] > 0.5).astype(int)
        weather_pipeline["humidity"] = 65  # Placeholder
        weather_pipeline.to_csv(OUTPUT_DIR / "weather_pipeline.csv", index=False)
    
    print(f"\n✅ All files saved to {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"  train.csv           — {len(train_df):,} rows ({len(CATEGORIES)} categories × {len(dates)} days)")
    print(f"  stores.csv          — 1 store (Dorchester bodega)")
    print(f"  holidays_events.csv — {len(holidays_df)} US holidays")
    print(f"  weather_boston.csv   — {len(weather) if weather is not None else 0} days of real Boston weather")
    print(f"  weather_pipeline.csv — Ready for feature engineering")
    
    print(f"\n📌 To use this data in any notebook, just change DATA_RAW to:")
    print(f'   DATA_RAW = Path(r"{OUTPUT_DIR}")')
    
    # Quick sanity check
    print(f"\nSanity check — Beverages sample:")
    bev = train_df[train_df.family == "BEVERAGES"].tail(7)
    for _, row in bev.iterrows():
        dt = pd.Timestamp(row["date"])
        print(f"  {dt.strftime('%A %b %d')}: {row['sales']} units" + 
              (" (promo)" if row["onpromotion"] else ""))