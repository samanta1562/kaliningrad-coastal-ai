import os
import json
import yaml
import duckdb
import numpy as np

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/silver") else "data_lake"
SILVER_DIR = os.path.join(BASE_DIR, "silver")
DUCKDB_PATH = os.path.join(BASE_DIR, "feature_store", "coastal_features.duckdb")

def test_named_shared_volume_in_compose():
    """Проверка явного объявления общего тома coastal_data_lake_vol в docker-compose.yml."""
    compose_path = "/app/docker-compose.yml" if os.path.exists("/app/docker-compose.yml") else "docker-compose.yml"
    if os.path.exists(compose_path):
        with open(compose_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "data_lake_vol" in cfg.get("volumes", {}), "Том data_lake_vol не объявлен в секции volumes"
        assert cfg["volumes"]["data_lake_vol"]["name"] == "coastal_data_lake_vol"

def test_silver_train_test_tensors_and_normalization():
    """Проверка 48 10-мерных тензоров (36 train / 12 test) и нормализации в диапазон [0, 1]."""
    manifest_path = os.path.join(SILVER_DIR, "silver_manifest.json")
    assert os.path.isfile(manifest_path), "Не найден silver_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["total_samples"] == 48, "Ожидалось 48 временных срезов (4 полигона x 12 эпох)"
    assert m["train_samples"] == 36, "Обучающая выборка должна содержать 36 сцен"
    assert m["test_samples"] == 12, "Тестовая выборка должна содержать 12 сцен"

    # Проверяем размерность и нормализацию первого тензора
    first_tensor_path = os.path.join(SILVER_DIR, "train", "curonian_root_2022_05_tensor10d.npy")
    arr = np.load(first_tensor_path)
    assert arr.shape == (10, 256, 256), f"Неверная форма тензора: {arr.shape}"
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0, "Тензор не нормализован в [0, 1]"

def test_duckdb_feature_store_populated():
    """Проверка аналитической таблицы в DuckDB Feature Store для обучения CatBoost."""
    assert os.path.isfile(DUCKDB_PATH), "База DuckDB Feature Store не создана"
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    total_rows = con.execute("SELECT COUNT(*) FROM coastal_feature_store").fetchone()[0]
    train_rows = con.execute("SELECT COUNT(*) FROM coastal_feature_store WHERE dataset_split = 'train'").fetchone()[0]
    test_rows = con.execute("SELECT COUNT(*) FROM coastal_feature_store WHERE dataset_split = 'test'").fetchone()[0]
    con.close()

    assert total_rows == 240, f"Ожидалось 240 записей в DuckDB, получено {total_rows}"
    assert train_rows == 180 and test_rows == 60


