import os
import json
import redis

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/gold") else "data_lake"
GOLD_DIR = os.path.join(BASE_DIR, "gold")
GOLD_REPORTS_DIR = os.path.join(GOLD_DIR, "reports")

REDIS_HOST = os.environ.get("REDIS_HOST", "redis-broker")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

def test_redis_geojson_layers_and_rbac_jwt_sessions():
    """Проверка наличия в памяти Redis 7.2 предрассчитанных GeoJSON-слоев (EPSG:4326), профилей TreeSHAP и JWT-сессий RBAC."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    assert r.ping() is True

    # 1. Проверяем GeoJSON-слой 48 береговых линий
    shorelines_raw = r.get("coastal:geojson:shorelines")
    assert shorelines_raw is not None, "Слой coastal:geojson:shorelines не найден в кэше Redis"
    shorelines_fc = json.loads(shorelines_raw)
    assert shorelines_fc["type"] == "FeatureCollection"
    assert len(shorelines_fc["features"]) == 48

    # Проверяем, что координаты переведены в градусы WGS84 (EPSG:4326) для веб-карты (долгота ~19.8..20.6, широта ~54.6..55.0)
    sample_coord = shorelines_fc["features"][0]["geometry"]["coordinates"][0][0]
    lon, lat = float(sample_coord[0]), float(sample_coord[1])
    assert 19.5 <= lon <= 21.0 and 54.4 <= lat <= 55.2, f"Координаты должны быть в WGS84 (EPSG:4326), получено: {lon}, {lat}"

    # 2. Проверяем GeoJSON-слой 20 трансектов с индексами риска CRI
    transects_fc = json.loads(r.get("coastal:geojson:transects_risk"))
    assert len(transects_fc["features"]) == 20
    assert r.ttl("coastal:geojson:transects_risk") > 0

    # 3. Проверяем наличие 3 ролевых JWT-сессий (etl_ingestor, ml_worker, ui_viewer)
    for role in ("etl_ingestor", "ml_worker", "ui_viewer"):
        jwt_raw = r.get(f"coastal:auth:jwt:{role}")
        assert jwt_raw is not None, f"Не найдена JWT-сессия для роли {role}"
        jwt_data = json.loads(jwt_raw)
        assert jwt_data["status"] == "ACTIVE"

    # 4. Проверяем отчет бенчмарка скорости
    bench_json = os.path.join(GOLD_DIR, "redis_benchmark_metrics.json")
    bench_png = os.path.join(GOLD_REPORTS_DIR, "visual_proof_redis_cache_benchmark.png")
    assert os.path.isfile(bench_json)
    assert os.path.isfile(bench_png) and os.path.getsize(bench_png) > 100_000

    with open(bench_json, "r", encoding="utf-8") as f:
        bm = json.load(f)
    assert bm["overall_mean_speedup_x"] >= 2.0, f"Кэш Redis должен быть минимум в 2 раза быстрее БД, получено: {bm['overall_mean_speedup_x']}x"


