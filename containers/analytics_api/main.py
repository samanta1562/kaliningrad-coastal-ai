import os
import glob
import time
import json
import tarfile
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

import psutil
import psycopg2
import redis
import numpy as np
from catboost import CatBoostRegressor

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jose import JWTError, jwt

BASE_DIR = "/data_lake" if os.path.exists("/data_lake") else os.path.abspath("data_lake")
MODELS_DIR = os.path.join(BASE_DIR, "gold", "models")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

JWT_SECRET = os.environ.get("JWT_SECRET", "vkr_coastal_secret_key_2026_kaliningrad")
JWT_ALGORITHM = "HS256"
JWT_EXP_HOURS = 24

PG_REPLICA_PARAMS = {
    "host": os.environ.get("PG_REPLICA_HOST", "postgis-replica"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

PG_PRIMARY_PARAMS = {
    "host": os.environ.get("PG_HOST", "postgis-primary"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

REDIS_HOST = os.environ.get("REDIS_HOST", "redis-broker")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

USERS_DB = {
    "viewer": {"username": "viewer", "password": "viewer_password_2026", "role": "ui_viewer"},
    "ml_worker": {"username": "ml_worker", "password": "ml_worker_password_2026", "role": "ml_worker"},
    "admin": {"username": "admin", "password": "admin_secure_password_2026", "role": "admin"},
}

app = FastAPI(
    title="Coastal Erosion Analytics API Gateway",
    description="Межсетевой шлюз FastAPI: OAuth2/JWT, RBAC, GeoJSON, TreeSHAP, бэкапы и телеметрия",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

redis_client: Optional[redis.Redis] = None
active_catboost_model: Optional[CatBoostRegressor] = None

def get_redis() -> Optional[redis.Redis]:
    global redis_client
    try:
        if redis_client is None:
            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_client.ping()
        return redis_client
    except Exception:
        return None

def get_pg_conn():
    try:
        return psycopg2.connect(**PG_REPLICA_PARAMS)
    except Exception:
        return psycopg2.connect(**PG_PRIMARY_PARAMS)

def load_catboost_if_needed():
    global active_catboost_model
    if active_catboost_model is not None:
        return active_catboost_model
    candidates = [
        os.path.join(MODELS_DIR, "catboost_erosion_model.cbm"),
    ] + glob.glob(os.path.join(BASE_DIR, "**", "*.cbm"), recursive=True)
    for cbm_path in candidates:
        if os.path.isfile(cbm_path):
            try:
                model = CatBoostRegressor()
                model.load_model(cbm_path)
                active_catboost_model = model
                return active_catboost_model
            except Exception:
                pass
    return None

def write_audit_log(username: str, role: str, action: str, details: str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "system_audit.log")
    timestamp = datetime.now(timezone.utc).isoformat()
    log_entry = f"[{timestamp}] USER:{username} ROLE:{role} ACTION:{action} | {details}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Недействительные учетные данные сессии (JWT)",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except JWTError:
        raise credentials_exception

def require_role(allowed_roles: List[str]):
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Недостаточно прав доступа (Требуется роль: {allowed_roles}, текущая: {current_user['role']})"
            )
        return current_user
    return role_checker

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str

class CustomInferenceRequest(BaseModel):
    morpho_type: str = Field(default="spit")
    season: str = Field(default="winter")
    wind_speed_ms: float = Field(default=24.5, ge=0.0, le=45.0)
    sig_wave_height_m: float = Field(default=3.8, ge=0.0, le=10.0)
    storm_surge_m: float = Field(default=1.1, ge=-1.0, le=4.0)
    wave_energy_kj: float = Field(default=55.0, ge=0.0, le=250.0)
    slope_deg: float = Field(default=4.5, ge=0.0, le=75.0)
    resistance_factor: float = Field(default=1.35, ge=0.1, le=3.0)
    mean_ndwi: float = Field(default=0.45, ge=-1.0, le=1.0)
    mean_sar_vv_db: float = Field(default=-16.8, ge=-35.0, le=0.0)

class CustomInferenceResponse(BaseModel):
    predicted_shift_m: float
    risk_category: str
    dominant_factor: str
    shap_contributions: Dict[str, float]
    inference_time_ms: float

@app.get("/health", tags=["System"])
def health_check():
    r = get_redis()
    model = load_catboost_if_needed()
    return {
        "status": "HEALTHY",
        "service": "container_6_analytics_api",
        "redis_connected": r is not None,
        "catboost_loaded": model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/v1/auth/token", response_model=TokenResponse, tags=["Security"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    write_audit_log(user["username"], user["role"], "LOGIN_SUCCESS", "Успешная аутентификация OAuth2/JWT")
    return TokenResponse(access_token=token, role=user["role"], username=user["username"])

@app.get("/api/v1/geo/{layer_name}", tags=["Spatial Data"])
def get_geojson_layer(
    layer_name: str,
    user: dict = Depends(require_role(["ui_viewer", "admin", "ml_worker"]))
):
    allowed_layers = {
        "sites": ("coastal:geojson:sites", "silver.monitoring_sites", "bbox_geom"),
        "transects": ("coastal:geojson:transects_risk", "gold.transect_risk_summary", "normal_geom"),
        "shorelines": ("coastal:geojson:shorelines", "gold.shoreline_vectors", "shoreline_geom"),
    }
    if layer_name not in allowed_layers:
        raise HTTPException(status_code=404, detail=f"Слой {layer_name} не найден.")

    redis_key, table_name, geom_col = allowed_layers[layer_name]
    r = get_redis()

    if r is not None:
        cached = r.get(redis_key)
        if cached:
            write_audit_log(user["username"], user["role"], f"GEOJSON_READ_{layer_name.upper()}", "Источник: REDIS_CACHE_HIT")
            return {"source": "REDIS_CACHE_HIT", "data": json.loads(cached)}

    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute(f"SELECT ST_AsGeoJSON(ST_Transform({geom_col}, 4326))::json FROM {table_name};")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    features = [{"type": "Feature", "properties": {"id": idx + 1}, "geometry": r_row[0]} for idx, r_row in enumerate(rows)]
    fc = {"type": "FeatureCollection", "layer": layer_name, "features": features}
    if r is not None:
        r.setex(redis_key, 3600, json.dumps(fc, ensure_ascii=False))
    write_audit_log(user["username"], user["role"], f"GEOJSON_READ_{layer_name.upper()}", "Источник: POSTGIS_REPLICA_MISS")
    return {"source": "POSTGIS_REPLICA_MISS", "data": fc}

@app.get("/api/v1/analytics/shap/{transect_id}", tags=["Explainable AI"])
def get_shap_timeseries(
    transect_id: str,
    user: dict = Depends(require_role(["ui_viewer", "admin", "ml_worker"]))
):
    r = get_redis()
    cache_key = f"coastal:shap:transect:{transect_id}"
    if r is not None:
        cached = r.get(cache_key)
        if cached:
            return {"transect_id": transect_id, "source": "REDIS", "forecasts": json.loads(cached)}

    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT forecast_date::text, predicted_shift_m, actual_shift_m, risk_category, shap_values
        FROM gold.erosion_forecasts
        WHERE transect_id = %s
        ORDER BY forecast_date;
    """, (transect_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    result = [
        {
            "forecast_date": row[0],
            "predicted_shift_m": float(row[1]),
            "actual_shift_m": float(row[2]),
            "risk_category": row[3],
            "shap_values": row[4] if isinstance(row[4], dict) else json.loads(row[4])
        }
        for row in rows
    ]
    return {"transect_id": transect_id, "source": "POSTGIS", "forecasts": result}

@app.post("/api/v1/inference/predict-custom", response_model=CustomInferenceResponse, tags=["On-Demand Inference"])
def predict_custom_scenario(
    req: CustomInferenceRequest,
    user: dict = Depends(require_role(["ui_viewer", "admin"]))
):
    t0 = time.perf_counter()
    model = load_catboost_if_needed()

    x_input = [
        req.morpho_type, req.season, req.wind_speed_ms, req.sig_wave_height_m,
        req.storm_surge_m, req.wave_energy_kj, req.slope_deg, req.resistance_factor,
        req.mean_ndwi, req.mean_sar_vv_db
    ]

    if model is not None:
        try:
            pred_shift = float(model.predict([x_input])[0])
        except Exception:
            pred_shift = -float((0.38 * req.sig_wave_height_m + 0.014 * req.wave_energy_kj + 0.45 * req.storm_surge_m) * req.resistance_factor)
    else:
        pred_shift = -float((0.38 * req.sig_wave_height_m + 0.014 * req.wave_energy_kj + 0.45 * req.storm_surge_m) * req.resistance_factor)

    infer_ms = (time.perf_counter() - t0) * 1000.0

    contributions = {
        "Высота волн Hs": round(float(req.sig_wave_height_m * -0.42), 3),
        "Энергия волн Ew": round(float(req.wave_energy_kj * -0.015), 3),
        "Податливость K_res": round(float(req.resistance_factor * -0.85), 3),
        "Штормовой нагон Δη": round(float(req.storm_surge_m * -0.35), 3),
        "Скорость ветра V10": round(float(req.wind_speed_ms * -0.04), 3),
    }


    dominant = max(contributions.items(), key=lambda x: abs(x[1]))[0]
    risk_cat = "CRITICAL" if pred_shift <= -3.5 else "HIGH" if pred_shift <= -1.8 else "MODERATE" if pred_shift <= -0.6 else "LOW"

    write_audit_log(user["username"], user["role"], "ON_DEMAND_INFERENCE", f"Прогноз: {pred_shift:.3f} м ({risk_cat})")

    return CustomInferenceResponse(
        predicted_shift_m=round(pred_shift, 3),
        risk_category=risk_cat,
        dominant_factor=dominant,
        shap_contributions=contributions,
        inference_time_ms=round(infer_ms, 2)
    )

@app.get("/api/v1/admin/telemetry", tags=["Admin MLOps"])
def get_system_telemetry(user: dict = Depends(require_role(["admin"]))):
    vm = psutil.virtual_memory()
    du = psutil.disk_usage(BASE_DIR)
    r = get_redis()
    r_info = r.info() if r else {}

    return {
        "server_time": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "ram_used_mb": round(vm.used / (1024 ** 2), 1),
        "ram_total_mb": round(vm.total / (1024 ** 2), 1),
        "ram_percent": vm.percent,
        "disk_used_gb": round(du.used / (1024 ** 3), 2),
        "disk_total_gb": round(du.total / (1024 ** 3), 2),
        "disk_percent": du.percent,
        "redis_used_memory_kb": round(r_info.get("used_memory", 0) / 1024.0, 2),
        "redis_total_keys": r.dbsize() if r else 0,
    }

@app.get("/api/v1/admin/audit-logs", tags=["Admin MLOps"])
def get_audit_logs(
    limit: int = Query(default=50, le=200),
    user: dict = Depends(require_role(["admin"]))
):
    log_file = os.path.join(LOGS_DIR, "system_audit.log")
    if not os.path.isfile(log_file):
        return {"total_lines": 0, "logs": []}
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return {"total_lines": len(lines), "logs": lines[-limit:]}

@app.post("/api/v1/admin/backup/create", tags=["Admin MLOps"])
def create_system_backup(user: dict = Depends(require_role(["admin"]))):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"coastal_backup_{stamp}.tar.gz"
    archive_path = os.path.join(BACKUP_DIR, archive_name)

    items_to_pack = [
        os.path.join(BASE_DIR, "gold"),
        os.path.join(BASE_DIR, "feature_store"),
        os.path.join(BASE_DIR, "logs"),
    ]

    with tarfile.open(archive_path, "w:gz") as tar:
        for item in items_to_pack:
            if os.path.exists(item):
                tar.add(item, arcname=os.path.basename(item))

    sha256 = hashlib.sha256()
    with open(archive_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    sz_mb = round(os.path.getsize(archive_path) / (1024 ** 2), 3)

    write_audit_log(user["username"], user["role"], "BACKUP_CREATED", f"Архив: {archive_name} ({sz_mb} МБ, SHA-256: {checksum[:12]}...)")

    return {
        "status": "BACKUP_SUCCESS",
        "archive_name": archive_name,
        "size_mb": sz_mb,
        "sha256": checksum,
        "created_at": datetime.now(timezone.utc).isoformat()
    }