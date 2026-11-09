-- Подключаем геоинформационное расширение PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- Создаем изолированные схемы медальонной архитектуры Data Lakehouse
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- ============================================================================
-- 1. СЛОЙ BRONZE: Неизменяемая зона приземления и аудита (Landing Zone)
-- ============================================================================
CREATE TABLE IF NOT EXISTS bronze.raw_ingestion_log (
    ingestion_id BIGSERIAL PRIMARY KEY,
    site_code VARCHAR(32) NOT NULL,
    source_type VARCHAR(32) NOT NULL,
    scene_or_batch_id VARCHAR(128) NOT NULL,
    acquisition_date DATE NOT NULL,
    file_path TEXT NOT NULL,
    sha256_checksum CHAR(64) UNIQUE NOT NULL,
    crs_epsg VARCHAR(16) DEFAULT 'EPSG:32634',
    raw_metadata JSONB NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bronze_raw_meta_gin 
ON bronze.raw_ingestion_log USING GIN (raw_metadata);

-- ============================================================================
-- 2. СЛОЙ SILVER: Очищенные пространственные и табличные сущности в 3НФ
-- ============================================================================

-- 2.1. Справочник береговых полигонов мониторинга (3НФ, геометрия Polygon)
CREATE TABLE IF NOT EXISTS silver.monitoring_sites (
    site_id INTEGER PRIMARY KEY,
    site_code VARCHAR(32) UNIQUE NOT NULL,
    site_name VARCHAR(128) NOT NULL,
    morpho_type VARCHAR(64) NOT NULL,
    resistance_coeff REAL NOT NULL CHECK (resistance_coeff > 0),
    center_lat NUMERIC(8,5) NOT NULL,
    center_lon NUMERIC(8,5) NOT NULL,
    bbox_geom GEOMETRY(Polygon, 32634) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_silver_sites_geom 
ON silver.monitoring_sites USING GIST (bbox_geom);

-- 2.2. Справочник береговых нормалей / трансектов (20 шт., геометрия LineString)
CREATE TABLE IF NOT EXISTS silver.coastal_transects (
    transect_id VARCHAR(32) PRIMARY KEY,
    site_id INTEGER NOT NULL REFERENCES silver.monitoring_sites(site_id) ON DELETE CASCADE,
    transect_num SMALLINT NOT NULL CHECK (transect_num BETWEEN 1 AND 50),
    azimuth_deg REAL NOT NULL CHECK (azimuth_deg BETWEEN 0 AND 360),
    mean_slope_deg REAL NOT NULL CHECK (mean_slope_deg >= 0),
    normal_geom GEOMETRY(LineString, 32634) NOT NULL,
    UNIQUE (site_id, transect_num)
);

CREATE INDEX IF NOT EXISTS idx_silver_transects_geom 
ON silver.coastal_transects USING GIST (normal_geom);

-- 2.3. Очищенные временные ряды гидрометеорологии ERA5
CREATE TABLE IF NOT EXISTS silver.era5_timeseries_clean (
    obs_id BIGSERIAL PRIMARY KEY,
    site_id INTEGER NOT NULL REFERENCES silver.monitoring_sites(site_id) ON DELETE CASCADE,
    obs_timestamp TIMESTAMPTZ NOT NULL,
    wind_speed_ms REAL NOT NULL CHECK (wind_speed_ms BETWEEN 0 AND 50),
    wind_dir_deg REAL NOT NULL CHECK (wind_dir_deg BETWEEN 0 AND 360),
    sig_wave_height_m REAL NOT NULL CHECK (sig_wave_height_m BETWEEN 0 AND 15),
    storm_surge_m REAL NOT NULL CHECK (storm_surge_m BETWEEN -2.0 AND 4.0),
    pressure_hpa REAL NOT NULL CHECK (pressure_hpa BETWEEN 850 AND 1080),
    UNIQUE (site_id, obs_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_silver_era5_site_time 
ON silver.era5_timeseries_clean (site_id, obs_timestamp);

-- 2.4. Реестр нормализованных 10D-тензоров со сквозной ссылкой на Bronze (Data Lineage)
CREATE TABLE IF NOT EXISTS silver.ml_samples_registry (
    sample_id VARCHAR(64) PRIMARY KEY,
    site_id INTEGER NOT NULL REFERENCES silver.monitoring_sites(site_id) ON DELETE CASCADE,
    ingestion_id BIGINT REFERENCES bronze.raw_ingestion_log(ingestion_id) ON DELETE SET NULL,
    epoch_date DATE NOT NULL,
    dataset_split VARCHAR(8) NOT NULL CHECK (dataset_split IN ('train', 'test')),
    cloud_cover_pct REAL NOT NULL CHECK (cloud_cover_pct BETWEEN 0 AND 100),
    processing_mode VARCHAR(32) NOT NULL,
    tensor_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    land_ratio REAL NOT NULL CHECK (land_ratio BETWEEN 0 AND 1)
);

-- ============================================================================
-- 3. СЛОЙ GOLD: Витрина результатов ИИ (Attention U-Net и CatBoost + TreeSHAP)
-- ============================================================================

-- 3.1. Реконструированные векторные полилинии уреза воды (выход U-Net)
CREATE TABLE IF NOT EXISTS gold.shoreline_vectors (
    shoreline_id SERIAL PRIMARY KEY,
    sample_id VARCHAR(64) NOT NULL REFERENCES silver.ml_samples_registry(sample_id) ON DELETE CASCADE,
    site_id INTEGER NOT NULL REFERENCES silver.monitoring_sites(site_id) ON DELETE CASCADE,
    observation_date DATE NOT NULL,
    dice_score REAL CHECK (dice_score BETWEEN 0 AND 1),
    iou_score REAL CHECK (iou_score BETWEEN 0 AND 1),
    shoreline_geom GEOMETRY(MultiLineString, 32634) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gold_shoreline_geom 
ON gold.shoreline_vectors USING GIST (shoreline_geom);

-- 3.2. Прогнозы размыва берега и векторы объяснимого ИИ Шепли (выход CatBoost)
CREATE TABLE IF NOT EXISTS gold.erosion_forecasts (
    forecast_id SERIAL PRIMARY KEY,
    transect_id VARCHAR(32) NOT NULL REFERENCES silver.coastal_transects(transect_id) ON DELETE CASCADE,
    sample_id VARCHAR(64) NOT NULL REFERENCES silver.ml_samples_registry(sample_id) ON DELETE CASCADE,
    forecast_date DATE NOT NULL,
    predicted_shift_m REAL NOT NULL,
    actual_shift_m REAL,
    risk_category VARCHAR(16) NOT NULL CHECK (risk_category IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')),
    shap_values JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gold_shap_gin 
ON gold.erosion_forecasts USING GIN (shap_values);
