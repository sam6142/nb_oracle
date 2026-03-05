"""
Model training, saving, and loading.

This is the ONLY place where models get trained and saved.
The dashboard and notebooks should use load_model(), never train directly.

Usage:
    # Train and save (run from terminal or cron job)
    python -m model.train

    # Load in dashboard or notebook
    from model.train import load_model, load_model_metadata
    model, metadata = load_model("BEVERAGES")
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    get_dataset_path,
    get_model_params,
    MODEL_REGISTRY,
)
from feature_store.engineer import build_features, get_feature_columns
from feature_store.sources.events import generate_boston_events, events_to_features
from model.evaluate import compute_wmape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_and_save(category, test_days=30):
    """
    Train a model for one category and save it to the registry.
    
    Saves two files:
        model_CATEGORY_TIMESTAMP.joblib  — the trained model
        model_CATEGORY_TIMESTAMP_meta.json — metadata (params, metrics, features)
    
    Also saves a "latest" pointer so loading always finds the newest model.
    
    Returns:
        dict with model, metadata, and evaluation results
    """
    logger.info(f"Training model for {category}...")
    
    # ── Load data ────────────────────────────────────────────
    data_path = get_dataset_path()
    
    train_df = pd.read_csv(data_path / "train.csv", parse_dates=["date"])
    holidays = pd.read_csv(data_path / "holidays_events.csv", parse_dates=["date"])
    
    weather_path = data_path / "weather_pipeline.csv"
    weather = pd.read_csv(weather_path, parse_dates=["date"]) if weather_path.exists() else None
    
    events_path = data_path / "events_features.csv"
    events = pd.read_csv(events_path, parse_dates=["date"]) if events_path.exists() else None
    
    # ── Filter to category ───────────────────────────────────
    store_data = train_df[train_df.store_nbr == train_df.store_nbr.iloc[0]]
    cat_data = (store_data[store_data.family == category]
                .sort_values("date").reset_index(drop=True))
    
    if len(cat_data) < 100:
        logger.warning(f"Not enough data for {category}: {len(cat_data)} rows")
        return None
    
    # ── Build features ───────────────────────────────────────
    features = build_features(cat_data, holidays,
                               weather_df=weather, events_df=events).dropna()
    feat_cols = get_feature_columns(features)
    
    # ── Split ────────────────────────────────────────────────
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    split_date = (pd.Timestamp(yesterday) - pd.Timedelta(days=test_days)).strftime("%Y-%m-%d")
    
    train_data = features[features.index < split_date]
    test_data = features[features.index >= split_date]
    
    if len(train_data) < 50 or len(test_data) < 3:
        logger.warning(f"Not enough data after split for {category}")
        return None
    
    X_train = train_data[feat_cols]
    y_train = train_data["sales"]
    X_test = test_data[feat_cols]
    y_test = test_data["sales"]
    
    # ── Train ────────────────────────────────────────────────
    params = get_model_params(category)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # ── Evaluate ─────────────────────────────────────────────
    predictions = np.maximum(model.predict(X_test), 0)
    wmape = compute_wmape(y_test.values, predictions)
    
    baseline_pred = test_data["sales_same_dow_avg_4w"].fillna(0).values
    baseline_wmape = compute_wmape(y_test.values, baseline_pred)
    
    improvement = (baseline_wmape - wmape) / baseline_wmape * 100 if baseline_wmape > 0 else 0
    
    logger.info(f"  WMAPE: {wmape:.1%} (baseline: {baseline_wmape:.1%}, improvement: {improvement:.1f}%)")
    
    # ── Save model ───────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_category = category.lower().replace(" ", "_").replace(",", "").replace("/", "_")
    
    model_filename = f"model_{safe_category}_{timestamp}.joblib"
    meta_filename = f"model_{safe_category}_{timestamp}_meta.json"
    
    model_path = MODEL_REGISTRY / model_filename
    meta_path = MODEL_REGISTRY / meta_filename
    
    joblib.dump(model, model_path)
    
    metadata = {
        "category": category,
        "version": timestamp,
        "model_file": model_filename,
        "params": params,
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "wmape": float(wmape),
        "baseline_wmape": float(baseline_wmape),
        "improvement_pct": float(improvement),
        "split_date": split_date,
        "trained_at": datetime.now().isoformat(),
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # ── Save "latest" pointer ────────────────────────────────
    latest_path = MODEL_REGISTRY / f"latest_{safe_category}.json"
    with open(latest_path, "w") as f:
        json.dump({
            "model_file": model_filename,
            "meta_file": meta_filename,
            "category": category,
            "updated_at": datetime.now().isoformat(),
        }, f, indent=2)
    
    logger.info(f"  Saved: {model_path}")
    
    return {
        "model": model,
        "metadata": metadata,
        "feat_cols": feat_cols,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }


def train_all_categories(categories=None, test_days=30):
    """
    Train models for multiple categories.
    If categories is None, trains all categories found in the data.
    """
    data_path = get_dataset_path()
    train_df = pd.read_csv(data_path / "train.csv", parse_dates=["date"])
    
    if categories is None:
        categories = sorted(train_df.family.unique())
    
    results = {}
    for cat in categories:
        result = train_and_save(cat, test_days=test_days)
        if result:
            results[cat] = result["metadata"]
    
    logger.info(f"\nTrained {len(results)}/{len(categories)} categories successfully")
    return results


# ══════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════

def load_model(category):
    """
    Load the latest trained model for a category.
    
    Returns:
        (model, metadata) tuple, or (None, None) if no model found
    """
    safe_category = category.lower().replace(" ", "_").replace(",", "").replace("/", "_")
    latest_path = MODEL_REGISTRY / f"latest_{safe_category}.json"
    
    if not latest_path.exists():
        logger.warning(f"No trained model found for {category}")
        return None, None
    
    with open(latest_path) as f:
        latest = json.load(f)
    
    model_path = MODEL_REGISTRY / latest["model_file"]
    meta_path = MODEL_REGISTRY / latest["meta_file"]
    
    if not model_path.exists():
        logger.warning(f"Model file missing: {model_path}")
        return None, None
    
    model = joblib.load(model_path)
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded model for {category} (version: {metadata['version']}, WMAPE: {metadata['wmape']:.1%})")
    
    return model, metadata


def get_model_feature_columns(category):
    """Get the feature columns a saved model expects."""
    _, metadata = load_model(category)
    if metadata is None:
        return None
    return metadata["feature_columns"]


def list_trained_models():
    """List all trained models in the registry."""
    models = []
    for f in MODEL_REGISTRY.glob("latest_*.json"):
        with open(f) as fh:
            data = json.load(fh)
        
        meta_path = MODEL_REGISTRY / data["meta_file"]
        if meta_path.exists():
            with open(meta_path) as mh:
                meta = json.load(mh)
            models.append({
                "category": data["category"],
                "version": meta["version"],
                "wmape": meta["wmape"],
                "trained_at": meta["trained_at"],
            })
    
    return pd.DataFrame(models) if models else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Neighborhood Oracle models")
    parser.add_argument("--category", type=str, default=None,
                        help="Category to train (e.g., 'BEVERAGES'). If not set, trains all.")
    parser.add_argument("--test-days", type=int, default=30,
                        help="Number of days to hold out for testing")
    args = parser.parse_args()
    
    if args.category:
        result = train_and_save(args.category, test_days=args.test_days)
        if result:
            print(f"\n✅ {args.category}: WMAPE={result['metadata']['wmape']:.1%}")
    else:
        results = train_all_categories(test_days=args.test_days)
        print(f"\n✅ Trained {len(results)} models")
        for cat, meta in results.items():
            print(f"  {cat:<25s} WMAPE: {meta['wmape']:.1%}")