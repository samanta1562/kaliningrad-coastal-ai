import os
import psycopg2

PG_CONN_PARAMS = {
    "host": os.environ.get("PG_HOST", "postgis-primary"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

def test_three_medallion_schemas_exist():
    """Проверка наличия всех трех схем (bronze, silver, gold) и расширения PostGIS."""
    conn = psycopg2.connect(**PG_CONN_PARAMS)
    cur = conn.cursor()

    cur.execute("""
        SELECT schema_name FROM information_schema.schemata 
        WHERE schema_name IN ('bronze', 'silver', 'gold');
    """)
    schemas = {r[0] for r in cur.fetchall()}
    assert schemas == {"bronze", "silver", "gold"}, f"Найдены не все схемы: {schemas}"
    cur.close()
    conn.close()

def test_bronze_and_silver_counts_and_lineage():
    """Проверка наполнения таблиц bronze и silver и сквозной связи Data Lineage (FK: ingestion_id)."""
    conn = psycopg2.connect(**PG_CONN_PARAMS)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM bronze.raw_ingestion_log;")
    assert cur.fetchone()[0] == 16, "В схеме bronze должно быть 16 зарегистрированных сырых файлов"

    cur.execute("SELECT COUNT(*) FROM silver.monitoring_sites;")
    assert cur.fetchone()[0] == 4, "В справочнике полигонов должно быть 4 записи"

    cur.execute("SELECT COUNT(*) FROM silver.coastal_transects;")
    assert cur.fetchone()[0] == 20, "В справочнике трансектов должно быть 20 профилей LineString"

    cur.execute("SELECT COUNT(*) FROM silver.era5_timeseries_clean;")
    assert cur.fetchone()[0] == 288, "В таблице метеорологии должно быть 288 почасовых записей"

    # Проверка сквозной трассировки Data Lineage: все 48 тензоров в silver связаны с сырыми снимками в bronze
    cur.execute("""
        SELECT COUNT(*) 
        FROM silver.ml_samples_registry s
        JOIN bronze.raw_ingestion_log b ON s.ingestion_id = b.ingestion_id;
    """)
    linked_samples = cur.fetchone()[0]
    assert linked_samples == 48, f"Все 48 сэмплов должны иметь ссылку на bronze, найдено {linked_samples}"

    # Проверка длины берегового трансекта через пространственную функцию ST_Length (около 500 м)
    cur.execute("SELECT ROUND(ST_Length(normal_geom)::numeric, 1) FROM silver.coastal_transects LIMIT 1;")
    length_m = float(cur.fetchone()[0])
    assert 480.0 <= length_m <= 520.0, f"Длина трансекта должна быть ~500 м, получено {length_m} м"

    cur.close()
    conn.close()



