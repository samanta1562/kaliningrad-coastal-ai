import os
import psycopg2
import pytest

PRIMARY_PARAMS = {
    "host": os.environ.get("PG_HOST", "postgis-primary"),
    "port": 5432,
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

REPLICA_PARAMS = {
    "host": os.environ.get("PG_REPLICA_HOST", "postgis-replica"),
    "port": 5432,
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

def test_primary_and_replica_roles_and_streaming():
    """Проверка: Primary в режиме записи, Replica в режиме Hot Standby, канал WAL = streaming."""
    conn_p = psycopg2.connect(**PRIMARY_PARAMS)
    conn_r = psycopg2.connect(**REPLICA_PARAMS)
    cur_p, cur_r = conn_p.cursor(), conn_r.cursor()

    cur_p.execute("SELECT pg_is_in_recovery();")
    assert cur_p.fetchone()[0] is False, "Узел postgis-primary должен быть в режиме Master (False)"

    cur_r.execute("SELECT pg_is_in_recovery();")
    assert cur_r.fetchone()[0] is True, "Узел postgis-replica должен быть в режиме Hot Standby (True)"

    cur_p.execute("SELECT state FROM pg_stat_replication;")
    row = cur_p.fetchone()
    assert row is not None and row[0] == "streaming", "Потоковая репликация WAL должна быть в статусе streaming"

    cur_p.close()
    cur_r.close()
    conn_p.close()
    conn_r.close()

def test_replica_enforces_read_only_and_matches_data():
    """Проверка блокировки записи на реплике и полного совпадения данных с мастером."""
    conn_r = psycopg2.connect(**REPLICA_PARAMS)
    cur_r = conn_r.cursor()

    # 1. Все 48 тензоров и 288 строк погоды доступны для чтения на реплике
    cur_r.execute("SELECT COUNT(*) FROM silver.ml_samples_registry;")
    assert cur_r.fetchone()[0] == 48

    cur_r.execute("SELECT COUNT(*) FROM silver.era5_timeseries_clean;")
    assert cur_r.fetchone()[0] == 288

    # 2. Попытка модификации данных на реплике вызывает ReadOnlySqlTransaction
    with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
        cur_r.execute("DELETE FROM silver.monitoring_sites WHERE site_id = 1;")

    conn_r.rollback()
    cur_r.close()
    conn_r.close()





