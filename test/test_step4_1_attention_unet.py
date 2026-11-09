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

def test_unet_weights_and_metrics_above_threshold():
    """Проверка сохранения весов .pt, графического отчета и превышения целевого порога качества (Dice >= 0.90, IoU >= 0.85)."""
    weights_file = os.path.join(GOLD_MODELS_DIR, "attention_unet_10d.pt")
    metrics_file = os.path.join(GOLD_MODELS_DIR, "unet_metrics.json")
    report_png = os.path.join(GOLD_REPORTS_DIR, "visual_proof_gold_unet_shorelines.png")

    assert os.path.isfile(weights_file), "Не найден файл весов attention_unet_10d.pt"
    assert os.path.isfile(metrics_file), "Не найден файл метрик unet_metrics.json"
    assert os.path.isfile(report_png) and os.path.getsize(report_png) > 100_000, "Визуальный отчет U-Net не создан"

    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["test_mean_dice"] >= 0.90, f"Метрика Dice ({m['test_mean_dice']}) ниже целевого порога 0.90"
    assert m["test_mean_iou"] >= 0.85, f"Метрика IoU ({m['test_mean_iou']}) ниже целевого порога 0.85"

def test_gold_shorelines_replicated_to_read_replica():
    """Проверка записи 48 полилиний берега в gold.shoreline_vectors и их доступности на реплике чтения (CQRS)."""
    conn = psycopg2.connect(**REPLICA_PARAMS)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), ROUND(AVG(dice_score)::numeric, 4) FROM gold.shoreline_vectors;")
    cnt, avg_dice = cur.fetchone()
    assert cnt == 48, f"В таблице gold.shoreline_vectors должно быть 48 векторных береговых линий, найдено {cnt}"
    assert float(avg_dice) >= 0.90

    # Проверяем, что геометрия валидна, имеет тип MultiLineString в проекции EPSG:32634 и ненулевую длину в метрах
    cur.execute("""
        SELECT ST_GeometryType(shoreline_geom), ST_SRID(shoreline_geom), ROUND(ST_Length(shoreline_geom)::numeric, 1)
        FROM gold.shoreline_vectors LIMIT 1;
    """)
    gtype, srid, length_m = cur.fetchone()
    assert gtype == "ST_MultiLineString"
    assert srid == 32634
    assert float(length_m) > 500.0, f"Длина береговой линии слишком мала: {length_m} м"

    cur.close()
    conn.close()


