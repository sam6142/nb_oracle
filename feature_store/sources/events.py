"""
Local events data source.

Events are categorized by their geographic impact on a specific store:

1. NEARBY (< 2 miles): Directly drives foot traffic to the store
   - Street fairs, local festivals, school events, church events
   - Impact: POSITIVE (more people walking by)

2. CITY-LEVEL (2-10 miles): Affects city traffic patterns
   - Major sports (Fenway, TD Garden), big concerts, conventions
   - Impact: MIXED (could draw people AWAY from your neighborhood)

3. UNIVERSAL: Affects all stores regardless of location
   - Holidays, Super Bowl, back-to-school — handled in calendar features
   - Not in this module (already in engineer.py as is_holiday, etc.)

Key principle: A Red Sox game at Fenway (3.2 miles from Dorchester)
might REDUCE your store's traffic as locals go to the game instead.
But a block party 0.5 miles away INCREASES it.
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


BOSTON_LAT = 42.3601
BOSTON_LON = -71.0589


def fetch_upcoming_events_ticketmaster(api_key, lat=BOSTON_LAT, lon=BOSTON_LON,
                                        radius_miles=10, days_ahead=7):
    """
    Fetch real upcoming events from Ticketmaster Discovery API.
    Free tier: 5,000 calls/day.
    Sign up at: https://developer.ticketmaster.com/
    """
    if not api_key:
        print("No Ticketmaster API key. Use generate_boston_events() instead.")
        return None

    url = "https://app.ticketmaster.com/discovery/v2/events.json"

    start = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "apikey": api_key,
        "latlong": f"{lat},{lon}",
        "radius": radius_miles,
        "unit": "miles",
        "startDateTime": start,
        "endDateTime": end,
        "size": 50,
        "sort": "date,asc",
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Ticketmaster API error: {response.status_code}")
        return None

    data = response.json()

    if "_embedded" not in data:
        return pd.DataFrame(columns=["date", "name", "venue", "category",
                                       "attendance_est", "distance_miles"])

    events = []
    for event in data["_embedded"]["events"]:
        event_date = event.get("dates", {}).get("start", {}).get("localDate")
        if not event_date:
            continue

        venue_name = ""
        if "_embedded" in event and "venues" in event["_embedded"]:
            venue_name = event["_embedded"]["venues"][0].get("name", "")

        category = "other"
        if "classifications" in event:
            segment = event["classifications"][0].get("segment", {}).get("name", "")
            if segment.lower() in ["sports", "music", "arts & theatre"]:
                category = segment.lower()

        attendance = 5000
        if "seatmap" in event:
            attendance = 15000

        events.append({
            "date": event_date,
            "name": event.get("name", "Unknown"),
            "venue": venue_name,
            "category": category,
            "attendance_est": attendance,
        })

    return pd.DataFrame(events)


def generate_boston_events(start_date, end_date, seed=42):
    """
    Generate realistic Boston-area events for demo purposes.
    No API key needed.

    Each event has a distance_miles field that determines its impact:
    - < 2 miles: nearby, directly boosts foot traffic
    - 2-5 miles: city-level, mixed impact (may draw people away)
    - > 5 miles: distant, minimal direct impact

    Based on real Boston venues and their distances from Dorchester.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date)

    events = []

    for date in dates:
        dt = pd.Timestamp(date)
        month = dt.month
        dow = dt.dayofweek

        # ── NEARBY EVENTS (< 2 miles from Dorchester) ───────
        # These BOOST store traffic

        # Dorchester community events (summer weekends)
        if 5 <= month <= 9 and dow >= 5:
            if np.random.random() < 0.20:
                event_types = [
                    ("Dorchester Block Party", 0.3, 500),
                    ("Fields Corner Street Fair", 0.5, 1500),
                    ("Dot Day Festival", 0.4, 3000),
                    ("Local Farmers Market", 0.6, 800),
                    ("Church Festival", 0.2, 400),
                ]
                name, dist, attend = event_types[np.random.randint(0, len(event_types))]
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "name": name,
                    "venue": "Dorchester",
                    "category": "community",
                    "attendance_est": attend + np.random.randint(-200, 200),
                    "distance_miles": dist,
                    "impact_type": "nearby_boost",
                })

        # School events (school year, weekdays)
        if (month >= 9 or month <= 6) and dow < 5:
            if np.random.random() < 0.08:
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "name": "School Event at Henderson",
                    "venue": "Henderson School",
                    "category": "community",
                    "attendance_est": np.random.randint(200, 600),
                    "distance_miles": 0.8,
                    "impact_type": "nearby_boost",
                })

        # ── CITY-LEVEL EVENTS (2-5 miles) ────────────────────
        # These have MIXED impact — may draw people AWAY

        # Red Sox at Fenway (3.2 miles from Dorchester)
        if 4 <= month <= 10:
            if np.random.random() < 0.45:
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "name": "Red Sox Home Game",
                    "venue": "Fenway Park",
                    "category": "sports",
                    "attendance_est": np.random.randint(30000, 38000),
                    "distance_miles": 3.2,
                    "impact_type": "city_draw_away",
                })

        # Celtics at TD Garden (4.5 miles)
        if month >= 10 or month <= 4:
            if np.random.random() < 0.35:
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "name": "Celtics Home Game",
                    "venue": "TD Garden",
                    "category": "sports",
                    "attendance_est": np.random.randint(18000, 20000),
                    "distance_miles": 4.5,
                    "impact_type": "city_draw_away",
                })

        # Bruins at TD Garden (4.5 miles)
        if month >= 10 or month <= 4:
            if np.random.random() < 0.35:
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "name": "Bruins Home Game",
                    "venue": "TD Garden",
                    "category": "sports",
                    "attendance_est": np.random.randint(17000, 18000),
                    "distance_miles": 4.5,
                    "impact_type": "city_draw_away",
                })

        # Concerts (various venues, various distances)
        concert_prob = 0.20 if 5 <= month <= 9 else 0.10
        if np.random.random() < concert_prob:
            venues = [
                ("TD Garden", 18000, 4.5, "city_draw_away"),
                ("MGM Music Hall", 5000, 4.3, "city_draw_away"),
                ("Roadrunner", 3500, 3.8, "city_draw_away"),
                ("House of Blues", 2500, 4.0, "city_draw_away"),
                ("Leader Bank Pavilion", 5000, 3.5, "city_draw_away"),
                ("Local Bar Dorchester", 150, 0.4, "nearby_boost"),
            ]
            venue, cap, dist, impact = venues[np.random.randint(0, len(venues))]
            events.append({
                "date": dt.strftime("%Y-%m-%d"),
                "name": f"Concert at {venue}",
                "venue": venue,
                "category": "music",
                "attendance_est": np.random.randint(int(cap * 0.6), cap),
                "distance_miles": dist,
                "impact_type": impact,
            })

        # ── SPECIAL EVENTS ───────────────────────────────────

        # Boston Marathon (3rd Monday of April, city-wide)
        if month == 4 and dow == 0 and 15 <= dt.day <= 21:
            events.append({
                "date": dt.strftime("%Y-%m-%d"),
                "name": "Boston Marathon",
                "venue": "Citywide",
                "category": "sports",
                "attendance_est": 500000,
                "distance_miles": 1.0,
                "impact_type": "nearby_boost",
            })

        # College events
        if (month >= 9 or month <= 5) and dow in [4, 5]:
            if np.random.random() < 0.20:
                colleges = [
                    ("UMass Boston Game", 1.5, "nearby_boost"),
                    ("BC Football", 5.0, "city_draw_away"),
                    ("BU Hockey", 4.0, "city_draw_away"),
                ]
                name, dist, impact = colleges[np.random.randint(0, len(colleges))]
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "name": name,
                    "venue": "Campus",
                    "category": "sports",
                    "attendance_est": np.random.randint(2000, 8000),
                    "distance_miles": dist,
                    "impact_type": impact,
                })

    return pd.DataFrame(events)


def events_to_features(events_df, dates):
    """
    Convert raw events list into daily feature columns.

    Key insight: nearby events BOOST demand, city events may REDUCE it.

    Features created:
    - nearby_events: count of events within 2 miles
    - nearby_attendance: total attendance of nearby events
    - city_events: count of events 2-10 miles away
    - city_attendance: total attendance of city events
    - has_sports_nearby: sports event within 2 miles?
    - has_sports_city: sports event at city level? (may draw people away)
    - has_music: any music event?
    - is_marathon: Boston Marathon day?
    - event_score: net impact score (positive = good for store)
    """
    dates = pd.to_datetime(dates)

    empty_result = pd.DataFrame({
        "date": dates,
        "nearby_events": 0,
        "nearby_attendance": 0,
        "city_events": 0,
        "city_attendance": 0,
        "has_sports_nearby": 0,
        "has_sports_city": 0,
        "has_music": 0,
        "is_marathon": 0,
        "event_score": 0,
    })

    if events_df is None or events_df.empty:
        return empty_result

    events = events_df.copy()
    events["date"] = pd.to_datetime(events["date"])

    result_rows = []

    for date in dates:
        day_events = events[events["date"] == date]

        if day_events.empty:
            result_rows.append({
                "date": date,
                "nearby_events": 0,
                "nearby_attendance": 0,
                "city_events": 0,
                "city_attendance": 0,
                "has_sports_nearby": 0,
                "has_sports_city": 0,
                "has_music": 0,
                "is_marathon": 0,
                "event_score": 0,
            })
            continue

        # Split by distance
        nearby = day_events[day_events["distance_miles"] <= 2]
        city = day_events[(day_events["distance_miles"] > 2) & 
                          (day_events["distance_miles"] <= 10)]

        # Calculate event score
        # Nearby events add to score, city events subtract
        nearby_score = nearby["attendance_est"].sum() / 1000 if not nearby.empty else 0
        city_score = city["attendance_est"].sum() / 10000 if not city.empty else 0

        # Marathon is always a huge positive (people everywhere)
        is_marathon = int(day_events["name"].str.contains("Marathon", case=False).any())
        if is_marathon:
            nearby_score += 50

        # Net score: nearby boost minus city drain
        event_score = round(nearby_score - city_score * 0.3, 2)

        result_rows.append({
            "date": date,
            "nearby_events": len(nearby),
            "nearby_attendance": int(nearby["attendance_est"].sum()) if not nearby.empty else 0,
            "city_events": len(city),
            "city_attendance": int(city["attendance_est"].sum()) if not city.empty else 0,
            "has_sports_nearby": int((nearby["category"] == "sports").any()) if not nearby.empty else 0,
            "has_sports_city": int((city["category"] == "sports").any()) if not city.empty else 0,
            "has_music": int((day_events["category"] == "music").any()),
            "is_marathon": is_marathon,
            "event_score": event_score,
        })

    return pd.DataFrame(result_rows)