-- The Neighborhood Oracle — Database Schema
-- This is the blueprint for our database tables.
-- We'll run this later when we set up Supabase.

CREATE EXTENSION IF NOT EXISTS postgis;

-- Stores table: one row per store we're tracking
CREATE TABLE stores (
    store_id          SERIAL PRIMARY KEY,
    name              TEXT NOT NULL,
    owner_phone       TEXT,
    location          GEOGRAPHY(POINT, 4326),
    timezone          TEXT DEFAULT 'America/New_York',
    avg_daily_revenue NUMERIC(10,2),
    sku_count         INT,
    neighborhood      TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_stores_location ON stores USING GIST(location);

-- Product categories (e.g., "beverages", "snacks", "dairy")
CREATE TABLE sku_categories (
    category_id     SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    parent_category TEXT
);

-- Sales: one row per store + category + day
CREATE TABLE sales (
    sale_id       BIGSERIAL PRIMARY KEY,
    store_id      INT REFERENCES stores(store_id),
    category_id   INT REFERENCES sku_categories(category_id),
    sale_date     DATE NOT NULL,
    units_sold    INT NOT NULL,
    revenue       NUMERIC(10,2),
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sales_lookup ON sales(store_id, category_id, sale_date);

-- Inventory: how much stock a store has on a given day
CREATE TABLE inventory (
    inventory_id      BIGSERIAL PRIMARY KEY,
    store_id          INT REFERENCES stores(store_id),
    category_id       INT REFERENCES sku_categories(category_id),
    snapshot_date     DATE NOT NULL,
    units_on_hand     INT NOT NULL,
    reorder_lead_days INT DEFAULT 2,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_inventory_lookup ON inventory(store_id, category_id, snapshot_date);

-- Feature vectors: THE KEY TABLE
-- This is where all our features come together.
-- Training reads rows WHERE units_sold_actual IS NOT NULL
-- Inference reads rows WHERE units_sold_actual IS NULL (future dates)
CREATE TABLE feature_vectors (
    id                        BIGSERIAL PRIMARY KEY,
    store_id                  INT REFERENCES stores(store_id),
    category_id               INT REFERENCES sku_categories(category_id),
    target_date               DATE NOT NULL,

    units_sold_actual         INT,

    -- How much did this store sell recently?
    sales_lag_7d              NUMERIC(10,2),
    sales_lag_14d             NUMERIC(10,2),
    sales_lag_28d             NUMERIC(10,2),
    sales_lag_56d             NUMERIC(10,2),
    sales_same_dow_avg_4w     NUMERIC(10,2),
    sales_same_dow_avg_8w     NUMERIC(10,2),
    sales_trend_4w            NUMERIC(10,4),
    inventory_velocity_7d     NUMERIC(10,4),

    -- What day/time is it?
    day_of_week_sin           NUMERIC(6,4),
    day_of_week_cos           NUMERIC(6,4),
    month_sin                 NUMERIC(6,4),
    month_cos                 NUMERIC(6,4),
    is_weekend                BOOLEAN,
    is_payday                 BOOLEAN,
    is_holiday                BOOLEAN,
    days_to_holiday           INT,

    -- What's the weather?
    temp_high                 NUMERIC(5,1),
    temp_feels_like           NUMERIC(5,1),
    temp_delta_vs_yesterday   NUMERIC(5,1),
    temp_delta_vs_weekly_avg  NUMERIC(5,1),
    is_precipitation          BOOLEAN,
    weather_severity          INT DEFAULT 0,

    -- Any events nearby?
    event_500m_total_attendance  INT DEFAULT 0,
    event_1km_total_attendance   INT DEFAULT 0,
    event_2km_total_attendance   INT DEFAULT 0,
    event_500m_is_sports         BOOLEAN DEFAULT FALSE,
    event_500m_is_music          BOOLEAN DEFAULT FALSE,
    hours_until_nearest_event    INT,

    -- Store info
    store_avg_daily_revenue      NUMERIC(10,2),
    store_sku_count              INT,

    created_at                TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(store_id, category_id, target_date)
);

CREATE INDEX idx_fv_lookup ON feature_vectors(store_id, category_id, target_date);
CREATE INDEX idx_fv_training ON feature_vectors(target_date)
    WHERE units_sold_actual IS NOT NULL;

-- Predictions: what the model thinks will happen
CREATE TABLE predictions (
    prediction_id     BIGSERIAL PRIMARY KEY,
    store_id          INT REFERENCES stores(store_id),
    category_id       INT REFERENCES sku_categories(category_id),
    target_date       DATE NOT NULL,
    horizon_days      INT NOT NULL,
    predicted_units   NUMERIC(10,2),
    prediction_lower  NUMERIC(10,2),
    prediction_upper  NUMERIC(10,2),
    current_inventory INT,
    shortfall_units   NUMERIC(10,2),
    model_version     TEXT,
    shap_top_features JSONB,
    explanation_text  TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(store_id, category_id, target_date, horizon_days)
);

-- Alerts: messages sent to store owners
CREATE TABLE alerts (
    alert_id      BIGSERIAL PRIMARY KEY,
    store_id      INT REFERENCES stores(store_id),
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    alert_type    TEXT NOT NULL,
    severity      TEXT NOT NULL,
    message       TEXT NOT NULL,
    sent_via      TEXT,
    sent_at       TIMESTAMPTZ,
    acknowledged  BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Model performance tracking
CREATE TABLE model_performance (
    id            BIGSERIAL PRIMARY KEY,
    model_version TEXT NOT NULL,
    eval_date     DATE NOT NULL,
    horizon_days  INT NOT NULL,
    wmape         NUMERIC(8,4),
    rmse          NUMERIC(10,2),
    n_predictions INT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);