import os
import json
import psycopg2

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/gold") else "data_lake"
GOLD_MODELS_DIR = os.path.join(BASE_DIR, "gold", "models")
GOLD_REPORTS_DIR = os.path.join(BASE_DIR, "gold", "reports")

REPLICA_PARAMS = {
    "host": os.environ.get("PG_REPLICA_HOST", "postgis-replica"),
    "port": 5432,
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

def test_catboost_model_and_accuracy_metrics():
    """Проверка сохранения модели .cbm, визуального отчета и высокой точности на тесте (MAE <= 0.85 м, R2 >= 0.85)."""
    cbm_file = os.path.join(GOLD_MODELS_DIR, "catboost_erosion_model.cbm")
    metrics_file = os.path.join(GOLD_MODELS_DIR, "catboost_shap_metrics.json")
    report_png = os.path.join(GOLD_REPORTS_DIR, "visual_proof_gold_catboost_shap.png")

    assert os.path.isfile(cbm_file), "Файл модели catboost_erosion_model.cbm не найден"
    assert os.path.isfile(metrics_file), "Файл метрик catboost_shap_metrics.json не найден"
    assert os.path.isfile(report_png) and os.path.getsize(report_png) > 100_000, "График CatBoost + SHAP не создан"

    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["test_mae_m"] <= 0.85, f"Ошибка MAE ({m['test_mae_m']} м) превышает порог 0.85 м"
    assert m["test_r2"] >= 0.85, f"Коэффициент детерминации R2 ({m['test_r2']}) ниже порога 0.85"

def test_gold_forecasts_and_shap_additivity_on_replica():
    """
    Проверка 240 прогнозов на реплике чтения PostGIS и математического свойства аддитивности Шепли:
    predicted_shift_m == base_value_m + SUM(contributions_m).
    """
    conn = psycopg2.connect(**REPLICA_PARAMS)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM gold.erosion_forecasts;")
    assert cur.fetchone()[0] == 240, "В таблице gold.erosion_forecasts должно быть 240 записей (20 трансектов x 12 эпох)"

    cur.execute("""
        SELECT predicted_shift_m, shap_values
        FROM gold.erosion_forecasts
        WHERE risk_category = 'CRITICAL'
        LIMIT 1;
    """)
    row = cur.fetchone()
    assert row is not None, "Должны присутствовать прогнозы с категорией риска CRITICAL"

    pred_m = float(row[0])
    shap_data = row[1] if isinstance(row[1], dict) else json.loads(row[1])
    base_val = float(shap_data["base_value_m"])
    contribs = shap_data["contributions_m"]

    # Сумма базового значения и вкладов всех 10 признаков должна равняться прогнозу модели
    shap_sum = base_val + sum(float(v) for v in contribs.values())
    assert abs(pred_m - shap_sum) < 0.02, f"Нарушена аддитивность TreeSHAP: прогноз={pred_m}, сумма Шепли={shap_sum}"
    assert float(contribs["wave_energy_kj"]) < 0.0, "Вклад штормовой энергии волн в размыв должен быть отрицательным (< 0 м)"

    cur.close()
    conn.close()


