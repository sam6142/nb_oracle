"""
Weather data source.

For development: generates realistic simulated weather data.
For production: pulls from OpenWeatherMap API.

WHY SIMULATE?
Our training data is from Ecuador (2013-2017). Getting exact historical
weather for those dates/locations for free is hard. So we simulate
realistic weather patterns that follow seasonal cycles, then later
swap in real API calls when we have a real store.
"""
import numpy as np
import pandas as pd
import requests
from config import OPENWEATHER_API_KEY


def generate_simulated_weather(dates, city="Quito", seed=42):
    """
    Creates realistic weather data for a list of dates.
    
    The simulation includes:
    - Seasonal temperature cycle (warmer in certain months)
    - Random daily variation
    - Precipitation that's more likely in rainy months
    - Temperature feels-like based on humidity
    
    Parameters:
        dates: list or array of dates
        city: city name (affects base temperature)
        seed: random seed for reproducibility
    
    Returns:
        DataFrame with columns: date, temp_high, temp_feels_like,
        precipitation_mm, is_precipitation, humidity
    """
    np.random.seed(seed)
    dates = pd.to_datetime(dates)
    n = len(dates)
    
    # Quito, Ecuador sits near the equator at high altitude
    # Temperature is fairly stable year-round (15-22°C / 59-72°F)
    # But it has a rainy season (Oct-May) and dry season (Jun-Sep)
    
    # Base temperature with seasonal variation
    day_of_year = dates.dayofyear
    # Slight seasonal cycle (warmer Feb-May, cooler Jun-Sep)
    seasonal = 2 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    
    base_temp_c = 18  # Quito average in Celsius
    daily_noise = np.random.normal(0, 2.5, n)  # Random daily variation
    
    temp_high_c = base_temp_c + seasonal + daily_noise
    
    # Convert to Fahrenheit (since many store owners think in F)
    temp_high_f = temp_high_c * 9/5 + 32
    
    # Humidity (higher in rainy season)
    month = dates.month
    rainy_months = [10, 11, 12, 1, 2, 3, 4, 5]
    base_humidity = np.where(np.isin(month, rainy_months), 75, 55)
    humidity = base_humidity + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 20, 100)
    
    # Feels-like temperature (humidity makes it feel warmer)
    feels_like_f = temp_high_f + (humidity - 50) * 0.05
    
    # Precipitation
    # More likely in rainy months, less in dry months
    rain_probability = np.where(np.isin(month, rainy_months), 0.55, 0.15)
    is_rain = np.random.random(n) < rain_probability
    
    # Amount of rain (when it rains)
    precip_mm = np.where(
        is_rain,
        np.random.exponential(8, n),  # Exponential distribution for rain amounts
        0
    )
    precip_mm = np.round(precip_mm, 1)
    
    weather_df = pd.DataFrame({
        "date": dates,
        "temp_high": np.round(temp_high_f, 1),
        "temp_feels_like": np.round(feels_like_f, 1),
        "precipitation_mm": precip_mm,
        "is_precipitation": is_rain.astype(int),
        "humidity": np.round(humidity, 0),
    })
    
    return weather_df


def fetch_weather_from_api(lat, lon, date):
    """
    Fetch weather from OpenWeatherMap API.
    
    THIS IS FOR PRODUCTION USE.
    Requires OPENWEATHER_API_KEY in .env file.
    
    For historical data, use the One Call API 3.0 "timemachine" endpoint.
    For forecasts, use the standard forecast endpoint.
    
    Parameters:
        lat: latitude
        lon: longitude  
        date: date to fetch weather for
    
    Returns:
        dict with temp_high, temp_feels_like, precipitation_mm, etc.
    """
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_key_here":
        raise ValueError(
            "No OpenWeatherMap API key found. "
            "Sign up at https://openweathermap.org/api and add your key to .env"
        )
    
    # Current / forecast weather
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "imperial"  # Fahrenheit
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    return {
        "date": date,
        "temp_high": data["main"]["temp_max"],
        "temp_feels_like": data["main"]["feels_like"],
        "precipitation_mm": data.get("rain", {}).get("1h", 0) * 1,
        "is_precipitation": 1 if "rain" in data or "snow" in data else 0,
        "humidity": data["main"]["humidity"],
    }