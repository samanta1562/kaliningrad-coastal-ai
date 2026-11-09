import os
import json
import numpy as np
import rasterio

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/bronze") else "data_lake"
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
EXPECTED_SITES = ["curonian_root", "svetlogorsk_cliff", "yantarny_beach", "baltiysk_spit"]

def resolve_path(rel_path: str) -> str:
    """Находит файл как внутри контейнера (/data_lake/...), так и на хосте (data_lake/...)."""
    if os.path.exists(rel_path):
        return rel_path
    container_path = "/" + rel_path.lstrip("/")
    if os.path.exists(container_path):
        return container_path
    return os.path.join(BASE_DIR, rel_path.replace("data_lake/", "").lstrip("/"))

def test_bronze_manifest_and_visual_proof_exist():
    manifest_file = os.path.join(BRONZE_DIR, "manifest.json")
    png_file = os.path.join(BRONZE_DIR, "visual_proof_bronze_sources.png")
    assert os.path.isfile(manifest_file), f"Не найден {manifest_file}"
    assert os.path.isfile(png_file), f"Не найден {png_file}"
    assert os.path.getsize(png_file) > 100_000, "PNG-файл со снимками слишком мал"

def test_no_black_squares_and_valid_storm_physics():
    manifest_file = os.path.join(BRONZE_DIR, "manifest.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for site in EXPECTED_SITES:
        site_info = manifest["sites"][site]

        # 1. Проверяем Sentinel-2: 5 каналов, EPSG:32634 и отсутствие черных квадратов (нулей)
        s2_path = resolve_path(site_info["files"]["s2"]["path"])
        with rasterio.open(s2_path) as src:
            assert src.count == 5
            assert src.crs.to_string() == "EPSG:32634"
            red_band = src.read(3)
            zero_fraction = np.count_nonzero(red_band <= 0.001) / red_band.size
            assert zero_fraction < 0.02, f"На снимке {site} обнаружены черные пустые зоны!"

        # 2. Проверяем физическую адекватность ветра (м/с) и высоты волны (м)
        meteo_path = resolve_path(site_info["files"]["meteo"]["path"])
        with open(meteo_path, "r", encoding="utf-8") as mf:
            records = json.load(mf)["records"]
        max_wind = max(r["wind_speed_ms"] for r in records)
        max_wave = max(r["sig_wave_height_m"] for r in records)
        assert 4.0 <= max_wind <= 25.0, f"Некорректная скорость ветра: {max_wind} м/с"
        assert 0.5 <= max_wave <= 6.0, f"Некорректная высота волны: {max_wave} м"