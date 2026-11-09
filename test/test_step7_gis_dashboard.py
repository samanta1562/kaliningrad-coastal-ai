import os
import sys

sys.path.insert(0, "/app")
from src.app import api_login, api_predict_custom, load_ai_metrics, parse_audit_logs, BASE_DIR, LOGS_DIR

def test_admin_and_viewer_login_audit():
    res_v = api_login("viewer", "viewer_password_2026")
    assert res_v["ok"] is True
    assert res_v["data"]["role"] == "ui_viewer"

    res_a = api_login("admin", "admin_secure_password_2026")
    assert res_a["ok"] is True
    assert res_a["data"]["role"] == "admin"

    log_file = os.path.join(LOGS_DIR, "system_audit.log")
    assert os.path.isfile(log_file)
    df_logs = parse_audit_logs(open(log_file, "r", encoding="utf-8").readlines())
    assert len(df_logs) >= 2
    assert "admin" in df_logs["Пользователь (Кто зашел)"].values

def test_ai_metrics_and_custom_prediction():
    metrics = load_ai_metrics()
    assert metrics["catboost"]["test_r2"] >= 0.85
    assert metrics["unet"]["test_mean_dice"] >= 0.85

    token = api_login("admin", "admin_secure_password_2026")["data"]["access_token"]
    pred = api_predict_custom({
        "morpho_type": "spit", "season": "winter",
        "wind_speed_ms": 25.0, "sig_wave_height_m": 4.0,
        "storm_surge_m": 1.2, "wave_energy_kj": 60.0,
        "slope_deg": 4.2, "resistance_factor": 1.35,
        "mean_ndwi": 0.45, "mean_sar_vv_db": -17.0,
    }, token)
    assert pred["predicted_shift_m"] < 0.0
    assert pred["risk_category"] in ("MODERATE", "HIGH", "CRITICAL")

def test_visual_proof_admin_dashboard_exists():
    png_path = os.path.join(BASE_DIR, "gold", "reports", "visual_proof_admin_mlops_dashboard.png")
    assert os.path.isfile(png_path)
