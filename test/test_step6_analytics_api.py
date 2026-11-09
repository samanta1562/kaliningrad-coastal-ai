import os
from fastapi.testclient import TestClient
from src.main import app, BASE_DIR

client = TestClient(app)

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"

def test_rbac_auth_tokens():
    # 1. Авторизация под обычным пользователем (viewer)
    res = client.post("/api/v1/auth/token", data={"username": "viewer", "password": "viewer_password_2026"})
    assert res.status_code == 200
    viewer_token = res.json()["access_token"]
    assert res.json()["role"] == "ui_viewer"

    # 2. Авторизация под администратором (admin)
    res_adm = client.post("/api/v1/auth/token", data={"username": "admin", "password": "admin_secure_password_2026"})
    assert res_adm.status_code == 200
    assert res_adm.json()["role"] == "admin"

    # 3. Проверка разграничения прав: viewer получает 403 Forbidden при попытке вызвать админский бэкап
    res_forbidden = client.post(
        "/api/v1/admin/backup/create",
        headers={"Authorization": f"Bearer {viewer_token}"}
    )
    assert res_forbidden.status_code == 403

def test_geojson_spatial_endpoint():
    res_auth = client.post("/api/v1/auth/token", data={"username": "viewer", "password": "viewer_password_2026"})
    token = res_auth.json()["access_token"]

    res = client.get(
        "/api/v1/geo/transects",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    body = res.json()
    assert "data" in body and body["data"]["type"] == "FeatureCollection"
    assert len(body["data"]["features"]) == 20

def test_on_demand_custom_inference():
    res_auth = client.post("/api/v1/auth/token", data={"username": "viewer", "password": "viewer_password_2026"})
    token = res_auth.json()["access_token"]

    payload = {
        "morpho_type": "spit",
        "season": "winter",
        "wind_speed_ms": 26.0,
        "sig_wave_height_m": 4.2,
        "storm_surge_m": 1.3,
        "wave_energy_kj": 65.0,
        "slope_deg": 4.0,
        "resistance_factor": 1.35,
        "mean_ndwi": 0.45,
        "mean_sar_vv_db": -17.5
    }

    res = client.post(
        "/api/v1/inference/predict-custom",
        json=payload,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_shift_m"] < 0.0
    assert data["risk_category"] in ("MODERATE", "HIGH", "CRITICAL")

def test_admin_backup_and_audit():
    res_adm = client.post("/api/v1/auth/token", data={"username": "admin", "password": "admin_secure_password_2026"})
    token = res_adm.json()["access_token"]

    res_bkp = client.post(
        "/api/v1/admin/backup/create",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res_bkp.status_code == 200
    data = res_bkp.json()
    assert data["status"] == "BACKUP_SUCCESS"
    assert len(data["sha256"]) == 64

    res_log = client.get(
        "/api/v1/admin/audit-logs?limit=5",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res_log.status_code == 200
    assert any("BACKUP_CREATED" in l for l in res_log.json()["logs"])





