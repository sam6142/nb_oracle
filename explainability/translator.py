"""
SHAP → Plain English Translator

Turns raw SHAP values into explanations a store owner
would understand and trust. This is the core product feature.

Design principles:
1. Sound like a helpful friend, not a robot
2. Combine related features into one thought
3. Always explain what the numbers MEAN, not just what they ARE
4. End with a simple takeaway
5. Keep it to 2-3 sentences max — owners are busy
"""
import numpy as np
import pandas as pd
import shap


def get_shap_explanation(model, features_row, feature_names, date,
                          prediction, actual=None):
    """
    Generate a full plain English explanation for one prediction.
    
    Parameters:
        model: trained XGBoost model
        features_row: single row of features (pandas Series)
        feature_names: list of feature column names
        date: the date being predicted
        prediction: the predicted value
        actual: (optional) actual value, for accuracy context
    
    Returns:
        dict with:
            - summary: one-line takeaway
            - explanation: 2-3 sentence plain English explanation
            - factors: list of (feature_name, direction, plain_text) tuples
            - confidence: "high", "moderate", or "low"
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(features_row.values.reshape(1, -1))[0]
    shap_series = pd.Series(sv, index=feature_names)
    
    # Get top features sorted by absolute impact
    top_features = shap_series.abs().nlargest(5).index.tolist()
    
    # Extract feature values and SHAP values for top features
    feature_info = {}
    for feat in top_features:
        feature_info[feat] = {
            "value": features_row[feat],
            "shap": shap_series[feat],
            "direction": "up" if shap_series[feat] > 0 else "down",
        }
    
    # ── Build the narrative ──────────────────────────────────
    day_name = date.strftime("%A")
    sentences = []
    used_features = set()
    
    # --- Weekend / Day of week story ---
    _add_day_story(sentences, used_features, feature_info, day_name, date)
    
    # --- Recent sales momentum story ---
    _add_momentum_story(sentences, used_features, feature_info)
    
    # --- Special events story (holiday, payday, promotion) ---
    _add_special_events_story(sentences, used_features, feature_info, date)
    
    # --- Weather story ---
    _add_weather_story(sentences, used_features, feature_info)
    
    # --- Catch any remaining high-impact features ---
    _add_remaining_features(sentences, used_features, feature_info, top_features)
    
    # Limit to 3 sentences
    sentences = sentences[:3]
    
    # ── Build summary ────────────────────────────────────────
    summary = _build_summary(prediction, feature_info, day_name)
    
    # ── Determine confidence ─────────────────────────────────
    confidence = _assess_confidence(shap_series, top_features)
    
    # ── Build factors list ───────────────────────────────────
    factors = []
    for feat in top_features[:3]:
        info = feature_info[feat]
        factors.append({
            "feature": feat,
            "value": info["value"],
            "direction": info["direction"],
            "impact": info["shap"],
        })
    
    explanation = " ".join(sentences)
    
    return {
        "summary": summary,
        "explanation": explanation,
        "factors": factors,
        "confidence": confidence,
        "shap_series": shap_series,
    }


def _add_day_story(sentences, used, info, day_name, date):
    """Explain the day-of-week and weekend effects."""
    
    has_weekend = "is_weekend" in info
    has_dow = "day_of_week" in info
    
    if not has_weekend and not has_dow:
        return
    
    is_weekend = info.get("is_weekend", {}).get("value", 0) == 1
    dow_pushing_up = info.get("day_of_week", {}).get("direction") == "up"
    weekend_pushing_up = info.get("is_weekend", {}).get("direction") == "up"
    
    if is_weekend:
        if weekend_pushing_up:
            sentences.append(
                f"{day_name}s are one of your busiest days — "
                f"expect higher than average traffic."
            )
        else:
            sentences.append(
                f"Even though it's the weekend, {day_name}s "
                f"tend to be a bit slower for this category."
            )
    else:
        if has_dow and not dow_pushing_up:
            sentences.append(
                f"{day_name}s tend to be on the quieter side for this category."
            )
        elif has_dow and dow_pushing_up:
            sentences.append(
                f"{day_name}s are typically a solid day for this category."
            )
    
    used.add("is_weekend")
    used.add("day_of_week")


def _add_momentum_story(sentences, used, info):
    """Explain recent sales trends and averages."""
    
    momentum_features = [
        "sales_avg_7d", "sales_avg_14d", "sales_avg_28d",
        "sales_1w_ago", "sales_same_dow_avg_4w", "sales_trend"
    ]
    
    active = {f: info[f] for f in momentum_features if f in info and f not in used}
    
    if not active:
        return
    
    # Check overall momentum direction
    up_count = sum(1 for f in active.values() if f["direction"] == "up")
    down_count = len(active) - up_count
    
    if up_count > down_count:
        # Momentum is positive
        if "sales_avg_7d" in active:
            avg = active["sales_avg_7d"]["value"]
            sentences.append(
                f"Your sales have been running strong this week "
                f"(averaging {avg:,.0f}/day), which is pushing the forecast up."
            )
        elif "sales_same_dow_avg_4w" in active:
            avg = active["sales_same_dow_avg_4w"]["value"]
            sentences.append(
                f"This day of the week has been averaging {avg:,.0f} units "
                f"over the past month — a solid trend."
            )
        elif "sales_trend" in active:
            sentences.append(
                "Sales have been trending upward recently, "
                "so we're forecasting a bit higher than usual."
            )
    else:
        # Momentum is negative
        if "sales_avg_7d" in active:
            avg = active["sales_avg_7d"]["value"]
            sentences.append(
                f"Sales have been a bit softer this week "
                f"(averaging {avg:,.0f}/day), which is pulling the forecast down."
            )
        elif "sales_same_dow_avg_4w" in active:
            avg = active["sales_same_dow_avg_4w"]["value"]
            sentences.append(
                f"This day of the week has been averaging just {avg:,.0f} units "
                f"recently — a bit below where it used to be."
            )
        elif "sales_trend" in active:
            sentences.append(
                "Sales have been trending downward this week, "
                "so we're being a bit more conservative."
            )
    
    for f in momentum_features:
        used.add(f)


def _add_special_events_story(sentences, used, info, date):
    """Explain holiday, payday, and promotion effects."""
    
    specials = []
    
    if "is_holiday" in info and info["is_holiday"]["value"] == 1:
        specials.append("it's a holiday")
        used.add("is_holiday")
    
    if "is_payday" in info and info["is_payday"]["value"] == 1:
        specials.append("it's near payday (people tend to spend more)")
        used.add("is_payday")
    
    if "on_promotion" in info and info["on_promotion"]["value"] >= 1:
        specials.append("this category is running a promotion")
        used.add("on_promotion")
    
    if specials:
        if len(specials) == 1:
            sentences.append(f"Also worth noting: {specials[0]}.")
        elif len(specials) == 2:
            sentences.append(
                f"A couple of extra factors: {specials[0]} and {specials[1]}."
            )
        else:
            joined = ", ".join(specials[:-1]) + f", and {specials[-1]}"
            sentences.append(f"Several things working together here: {joined}.")


def _add_weather_story(sentences, used, info):
    """Explain weather effects."""
    
    weather_features = [
        "temp_high", "temp_feels_like", "temp_delta_vs_yesterday",
        "temp_delta_vs_weekly_avg", "is_precipitation", "is_hot_day",
        "is_cold_day", "weather_severity", "rain_streak_days",
        "precipitation_mm"
    ]
    
    active = {f: info[f] for f in weather_features if f in info and f not in used}
    
    if not active:
        return
    
    # Hot day
    if "is_hot_day" in active and active["is_hot_day"]["value"] == 1:
        sentences.append(
            "It's hotter than usual — cold drinks and frozen items "
            "typically see a bump on days like this."
        )
        for f in weather_features:
            used.add(f)
        return
    
    # Cold day
    if "is_cold_day" in active and active["is_cold_day"]["value"] == 1:
        sentences.append(
            "It's cooler than usual — warm comfort foods and "
            "hot beverages tend to pick up."
        )
        for f in weather_features:
            used.add(f)
        return
    
    # Big temperature change
    if "temp_delta_vs_yesterday" in active:
        delta = active["temp_delta_vs_yesterday"]["value"]
        if abs(delta) > 5:
            direction = "warmer" if delta > 0 else "cooler"
            sentences.append(
                f"Temperature is swinging {abs(delta):.0f}°F {direction} "
                f"compared to yesterday — sudden changes like this "
                f"often shift what people buy."
            )
            for f in weather_features:
                used.add(f)
            return
    
    # Rain
    if "is_precipitation" in active and active["is_precipitation"]["value"] == 1:
        sentences.append(
            "Rain is in the forecast — expect foot traffic to dip "
            "but basket sizes to go up as people stock up."
        )
        for f in weather_features:
            used.add(f)
        return
    
    for f in weather_features:
        used.add(f)


def _add_remaining_features(sentences, used, info, top_features):
    """Catch any top features we haven't explained yet."""
    
    remaining = [f for f in top_features[:3] if f not in used]
    
    for feat in remaining:
        val = info[feat]["value"]
        direction = info[feat]["direction"]
        
        # Generic but still readable
        clean_name = feat.replace("_", " ")
        effect = "pushing demand higher" if direction == "up" else "pulling demand lower"
        sentences.append(
            f"The {clean_name} signal ({val:.0f}) is also {effect}."
        )
        used.add(feat)


def _build_summary(prediction, feature_info, day_name):
    """One-line takeaway for the alert header."""
    
    # Count ups vs downs
    ups = sum(1 for f in feature_info.values() if f["direction"] == "up")
    downs = len(feature_info) - ups
    
    if ups > downs + 1:
        tone = "Looking like a strong"
    elif downs > ups + 1:
        tone = "Expecting a quieter"
    else:
        tone = "Shaping up to be a typical"
    
    return f"{tone} {day_name} — forecast is ~{prediction:,.0f} units."


def _assess_confidence(shap_series, top_features):
    """
    Determine confidence based on how much the top features agree.
    If they all push the same direction = high confidence.
    If they're split = low confidence.
    """
    top_shap = shap_series[top_features[:5]]
    positive = (top_shap > 0).sum()
    negative = (top_shap < 0).sum()
    
    agreement = max(positive, negative) / len(top_shap)
    
    if agreement >= 0.8:
        return "high"
    elif agreement >= 0.6:
        return "moderate"
    else:
        return "low"


def format_whatsapp_message(store_name, category, date, prediction,
                             explanation_result, current_inventory=None):
    """
    Format a complete WhatsApp/SMS message for a store owner.
    
    This is the final output — what lands on their phone.
    """
    summary = explanation_result["summary"]
    explanation = explanation_result["explanation"]
    confidence = explanation_result["confidence"]
    
    confidence_emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}
    conf_emoji = confidence_emoji.get(confidence, "⚪")
    
    msg = f"🔮 *{store_name} — {category}*\n"
    msg += f"📅 {date.strftime('%A, %B %d')}\n\n"
    msg += f"*{summary}*\n\n"
    msg += f"{explanation}\n\n"
    msg += f"{conf_emoji} Confidence: {confidence.capitalize()}\n"
    
    if current_inventory is not None:
        shortfall = current_inventory - prediction
        if shortfall < 0:
            msg += f"\n⚠️ *Heads up:* You have {current_inventory:,.0f} units "
            msg += f"on hand but we're forecasting {prediction:,.0f}. "
            msg += f"Consider ordering {abs(shortfall):,.0f} more."
        elif shortfall < prediction * 0.2:
            msg += f"\n📦 Stock is tight — you have {current_inventory:,.0f} "
            msg += f"units with only {shortfall:,.0f} units of buffer."
        else:
            msg += f"\n✅ Stock looks good — {current_inventory:,.0f} units "
            msg += f"on hand with {shortfall:,.0f} units of buffer."
    
    return msg