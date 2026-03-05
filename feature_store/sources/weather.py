"""
Weather data source.

Three modes:
1. generate_simulated_weather() — fake data for development
2. fetch_historical_weather() — real past weather from Open-Meteo (free)
3. fetch_weather_forecast() — real 7-day forecast from Open-Meteo (free)

No API key needed for any of these.
Coordinates come from config.py — no hardcoded values here.
"""
import numpy as np
import pandas as pd
import requests
from config import STORE_LAT, STORE_LON


def generate_simulated_weather(dates, seed=42):
    """Fake weather for development."""
    np.random.seed(seed)
    dates = pd.to_datetime(dates)
    n = len(dates)
    
    day_of_year = dates.dayofyear
    seasonal = 2 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    
    base_temp_c = 18
    daily_noise = np.random.normal(0, 2.5, n)
    temp_high_c = base_temp_c + seasonal + daily_noise
    temp_high_f = temp_high_c * 9/5 + 32
    
    month = dates.month
    rainy_months = [10, 11, 12, 1, 2, 3, 4, 5]
    base_humidity = np.where(np.isin(month, rainy_months), 75, 55)
    humidity = base_humidity + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 20, 100)
    
    feels_like_f = temp_high_f + (humidity - 50) * 0.05
    
    rain_probability = np.where(np.isin(month, rainy_months), 0.55, 0.15)
    is_rain = np.random.random(n) < rain_probability
    precip_mm = np.where(is_rain, np.random.exponential(8, n), 0)
    
    return pd.DataFrame({
        "date": dates,
        "temp_high": np.round(temp_high_f, 1),
        "temp_feels_like": np.round(feels_like_f, 1),
        "precipitation_mm": np.round(precip_mm, 1),
        "is_precipitation": is_rain.astype(int),
        "humidity": np.round(humidity, 0),
    })


def fetch_historical_weather(start_date, end_date, lat=None, lon=None):
    """
    Fetch REAL historical weather from Open-Meteo.
    Free, no API key needed. Works for any date from 1940 to yesterday.
    """
    lat = lat or STORE_LAT
    lon = lon or STORE_LON
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Weather API error: {response.status_code}")
        return None
    
    data = response.json()
    
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_high": data["daily"]["temperature_2m_max"],
        "temp_feels_like": data["daily"]["apparent_temperature_max"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
    }).fillna(0)
    
    df["is_precipitation"] = (df["precipitation_mm"] > 0.5).astype(int)
    df["humidity"] = 65
    
    return df


def fetch_weather_forecast(lat=None, lon=None):
    """
    Fetch REAL 7-day weather forecast from Open-Meteo.
    Free, no API key needed. Returns today + next 6 days.
    """
    lat = lat or STORE_LAT
    lon = lon or STORE_LON
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,precipitation_sum,precipitation_probability_max",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": 7,
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Forecast API error: {response.status_code}")
        return None
    
    data = response.json()
    
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_high": data["daily"]["temperature_2m_max"],
        "temp_feels_like": data["daily"]["apparent_temperature_max"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
        "rain_probability": data["daily"]["precipitation_probability_max"],
    }).fillna(0)
    
    df["is_precipitation"] = (df["precipitation_mm"] > 0.5).astype(int)
    df["humidity"] = 65
    
    return df