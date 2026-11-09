import os
import duckdb
import psycopg2

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/gold") else "data_lake"
DUCKDB_PATH = os.path.join(BASE_DIR, "feature_store", "coastal_features.duckdb")
GOLD_REPORTS_DIR = os.path.join(BASE_DIR, "gold", "reports")

REPLICA_PARAMS = {
    "host": os.environ.get("PG_REPLICA_HOST", "postgis-replica"),
    "port": 5432,
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}


def test_gold_risk_summary_published_in_postgis_replica_and_duckdb():
    """Проверка публикации витрины gold.transect_risk_summary в PostGIS (с репликацией на CQRS-узел) и в DuckDB."""
    # 1. Проверяем чтение новой витрины с реплики чтения PostGIS
    conn = psycopg2.connect(**REPLICA_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), MIN(cri_index), MAX(cri_index), COUNT(DISTINCT risk_grade)
        FROM gold.transect_risk_summary;
    """)
    cnt, min_cri, max_cri, grades_cnt = cur.fetchone()
    cur.close()
    conn.close()

    assert cnt == 20, f"В витрине gold.transect_risk_summary должно быть 20 трансектов, найдено {cnt}"
    assert 0.0 <= float(min_cri) < float(max_cri) <= 100.0
    assert grades_cnt >= 2, "Должна присутствовать дифференциация трансектов по категориям риска"

    # 2. Проверяем наличие зеркальной OLAP-витрины в DuckDB
    con_duck = duckdb.connect(DUCKDB_PATH, read_only=True)
    duck_cnt = con_duck.execute("SELECT COUNT(*) FROM gold_transect_risk_mart;").fetchone()[0]
    con_duck.close()
    assert duck_cnt == 20, "В DuckDB должна быть опубликована таблица gold_transect_risk_mart на 20 строк"

    # 3. Проверяем наличие графического отчета
    png_file = os.path.join(GOLD_REPORTS_DIR, "visual_proof_gold_aggregator_marts.png")
    assert os.path.isfile(png_file) and os.path.getsize(png_file) > 120_000
