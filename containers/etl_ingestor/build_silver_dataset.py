import os
import json
import duckdb
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, gaussian_gradient_magnitude
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/bronze") else os.path.abspath("data_lake")
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "silver")
FEATURE_STORE_DIR = os.path.join(BASE_DIR, "feature_store")
DUCKDB_PATH = os.path.join(FEATURE_STORE_DIR, "coastal_features.duckdb")

# 12 репрезентативных временных срезов (2022-2024 -> TRAIN [9 эпох], 2025 -> TEST [3 эпохи])
EPOCHS = [
    {"epoch_id": "2022_05", "date": "2022-05-18", "season": "spring", "cloud_pct": 5.0,  "wind_ms": 6.2,  "hs_m": 0.85, "surge_m": 0.12, "split": "train"},
    {"epoch_id": "2022_08", "date": "2022-08-14", "season": "summer", "cloud_pct": 3.0,  "wind_ms": 5.4,  "hs_m": 0.65, "surge_m": 0.05, "split": "train"},
    {"epoch_id": "2022_11", "date": "2022-11-21", "season": "autumn", "cloud_pct": 85.0, "wind_ms": 16.8, "hs_m": 3.40, "surge_m": 0.92, "split": "train"},
    {"epoch_id": "2023_02", "date": "2023-02-10", "season": "winter", "cloud_pct": 92.0, "wind_ms": 18.2, "hs_m": 3.85, "surge_m": 1.15, "split": "train"},
    {"epoch_id": "2023_06", "date": "2023-06-22", "season": "summer", "cloud_pct": 8.0,  "wind_ms": 6.0,  "hs_m": 0.75, "surge_m": 0.08, "split": "train"},
    {"epoch_id": "2023_10", "date": "2023-10-08", "season": "autumn", "cloud_pct": 78.0, "wind_ms": 15.5, "hs_m": 3.10, "surge_m": 0.84, "split": "train"},
    {"epoch_id": "2024_01", "date": "2024-01-15", "season": "winter", "cloud_pct": 88.0, "wind_ms": 17.4, "hs_m": 3.60, "surge_m": 1.02, "split": "train"},
    {"epoch_id": "2024_07", "date": "2024-07-19", "season": "summer", "cloud_pct": 4.0,  "wind_ms": 5.8,  "hs_m": 0.70, "surge_m": 0.06, "split": "train"},
    {"epoch_id": "2024_11", "date": "2024-11-16", "season": "autumn", "cloud_pct": 65.0, "wind_ms": 16.1, "hs_m": 3.25, "surge_m": 0.88, "split": "train"},
    {"epoch_id": "2025_02", "date": "2025-02-04", "season": "winter", "cloud_pct": 90.0, "wind_ms": 17.9, "hs_m": 3.75, "surge_m": 1.08, "split": "test"},
    {"epoch_id": "2025_05", "date": "2025-05-25", "season": "spring", "cloud_pct": 6.0,  "wind_ms": 6.5,  "hs_m": 0.90, "surge_m": 0.14, "split": "test"},
    {"epoch_id": "2025_09", "date": "2025-09-28", "season": "autumn", "cloud_pct": 12.0, "wind_ms": 13.8, "hs_m": 2.65, "surge_m": 0.68, "split": "test"},
]

SITE_RESISTANCE = {
    "curonian_root":     {"morpho": "sandy_spit",     "resistance_factor": 1.35, "mean_slope_deg": 4.2},
    "svetlogorsk_cliff": {"morpho": "abrasion_cliff", "resistance_factor": 0.55, "mean_slope_deg": 28.5},
    "yantarny_beach":    {"morpho": "wide_acc_beach", "resistance_factor": 0.40, "mean_slope_deg": 2.8},
    "baltiysk_spit":     {"morpho": "sandy_spit",     "resistance_factor": 1.45, "mean_slope_deg": 4.6},
}

def zscore_norm(arr: np.ndarray) -> np.ndarray:
    """Робастная Z-score стандартизация канала с усечением выбросов [-3, +3] и приведением к [0, 1]."""
    mean = float(np.mean(arr))
    std = float(np.std(arr)) + 1e-6
    z = np.clip((arr - mean) / std, -3.0, 3.0)
    return ((z + 3.0) / 6.0).astype(np.float32)

def build_10d_tensor_and_mask(s2: np.ndarray, s1: np.ndarray, dem: np.ndarray, epoch: dict, seed: int):
    """
    Формирование стандартизированного 10-канального тензора (10, 256, 256) и маски уреза воды:
    Каналы: [0:B2, 1:B3, 2:B4, 3:B8, 4:B11, 5:NDWI, 6:MNDWI, 7:SAR_VV, 8:SAR_VH, 9:DEM]
    """
    rng = np.random.default_rng(seed)
    # Легкая сезонная/погодная вариация отражательной способности и уровня нагона
    surge_shift = epoch["surge_m"] * 0.018
    b2 = np.clip(s2[0] + rng.normal(0, 0.004, s2[0].shape), 0.001, 1.0)
    b3 = np.clip(s2[1] + rng.normal(0, 0.004, s2[1].shape), 0.001, 1.0)
    b4 = np.clip(s2[2] + rng.normal(0, 0.004, s2[2].shape), 0.001, 1.0)
    b8 = np.clip(s2[3] - surge_shift + rng.normal(0, 0.005, s2[3].shape), 0.001, 1.0)
    b11 = np.clip(s2[4] - surge_shift + rng.normal(0, 0.005, s2[4].shape), 0.001, 1.0)

    # Спектральные водные индексы NDWI и MNDWI (диапазон [-1, 1] -> нормализация в [0, 1])
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    mndwi = (b3 - b11) / (b3 + b11 + 1e-6)

    # Сглаживание спекл-шума SAR (фильтр Ли / локальное среднее 3x3)
    vv_filtered = uniform_filter(s1[0] + rng.normal(0, 0.4, s1[0].shape), size=3)
    vh_filtered = uniform_filter(s1[1] + rng.normal(0, 0.4, s1[1].shape), size=3)

    # Эталонная бинарная маска: 1.0 = Суша, 0.0 = Водная акватория
    # Комбинирует спектральный индекс воды, альбедо NIR и превышение рельефа над штормовым нагоном
    water_score = (ndwi > 0.05) & (b8 < np.percentile(b8, 48) + surge_shift) & (dem < 2.5 + epoch["surge_m"])
    land_mask = (~water_score).astype(np.float32)

    # При сплошной облачности (>= 20%) оптические каналы зашумляются облаками,
    # чтобы нейросеть училась опираться на всепогодный радар Sentinel-1 и DEM
    if epoch["cloud_pct"] >= 20.0:
        cloud_veil = rng.uniform(0.35, 0.65, size=b2.shape).astype(np.float32)
        b2 = 0.5 * b2 + 0.5 * cloud_veil
        b3 = 0.5 * b3 + 0.5 * cloud_veil
        b4 = 0.5 * b4 + 0.5 * cloud_veil

    tensor_10d = np.stack([
        zscore_norm(b2),
        zscore_norm(b3),
        zscore_norm(b4),
        zscore_norm(b8),
        zscore_norm(b11),
        ((ndwi + 1.0) / 2.0).astype(np.float32),
        ((mndwi + 1.0) / 2.0).astype(np.float32),
        zscore_norm(vv_filtered),
        zscore_norm(vh_filtered),
        (np.clip(dem, 0.0, 50.0) / 50.0).astype(np.float32),
    ], axis=0)

    return tensor_10d, land_mask, ndwi, mndwi, vv_filtered

def main():
    os.makedirs(os.path.join(SILVER_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(SILVER_DIR, "test"), exist_ok=True)
    os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

    with open(os.path.join(BRONZE_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        bronze_manifest = json.load(f)

    silver_catalog = []
    transect_rows = []
    visual_samples = {}

    for site_code, site_info in bronze_manifest["sites"].items():
        s2_path = os.path.join(BRONZE_DIR, site_code, f"{site_code}_S2_L2A.tif")
        s1_path = os.path.join(BRONZE_DIR, site_code, f"{site_code}_S1_SAR.tif")
        dem_path = os.path.join(BRONZE_DIR, site_code, f"{site_code}_COP_DEM.tif")

        with rasterio.open(s2_path) as src:
            s2_base = src.read()
        with rasterio.open(s1_path) as src:
            s1_base = src.read()
        with rasterio.open(dem_path) as src:
            dem_base = src.read(1)

        slope_map = gaussian_gradient_magnitude(dem_base, sigma=1.5)
        res_meta = SITE_RESISTANCE[site_code]

        for ep_idx, ep in enumerate(EPOCHS):
            sample_id = f"{site_code}_{ep['epoch_id']}"
            split_dir = os.path.join(SILVER_DIR, ep["split"])

            tensor_10d, mask, ndwi, mndwi, vv_filt = build_10d_tensor_and_mask(
                s2_base, s1_base, dem_base, ep, seed=site_info["site_id"] * 100 + ep_idx
            )

            tensor_file = os.path.join(split_dir, f"{sample_id}_tensor10d.npy")
            mask_file = os.path.join(split_dir, f"{sample_id}_mask.npy")
            np.save(tensor_file, tensor_10d)
            np.save(mask_file, mask)

            mode = "SAR_ALL_WEATHER" if ep["cloud_pct"] >= 20.0 else "MULTIMODAL_OPTIC_SAR"
            silver_catalog.append({
                "sample_id": sample_id,
                "site_id": site_info["site_id"],
                "site_code": site_code,
                "epoch_date": ep["date"],
                "split": ep["split"],
                "cloud_cover_pct": ep["cloud_pct"],
                "processing_mode": mode,
                "tensor_shape": list(tensor_10d.shape),
                "tensor_path": f"data_lake/silver/{ep['split']}/{sample_id}_tensor10d.npy",
                "mask_path": f"data_lake/silver/{ep['split']}/{sample_id}_mask.npy",
                "land_ratio": round(float(np.mean(mask)), 4)
            })

            # Сохраняем последний тестовый срез каждого полигона для визуального отчета
            if ep["epoch_id"] == "2025_09":
                visual_samples[site_code] = {
                    "name": site_info["name"],
                    "ndwi": ndwi,
                    "mndwi": mndwi,
                    "vv_filt": vv_filt,
                    "mask": mask,
                    "tensor_sample": tensor_10d
                }

            # Генерируем 5 береговых трансектов на каждый полигон для DuckDB Feature Store (итого 4*5*12 = 240 строк)
            wave_energy = round(0.5 * 1025.0 * 9.81 * (ep["hs_m"] ** 2) / 1000.0, 3)  # кДж/м^2
            for t_num in range(1, 6):
                transect_id = f"{site_code}_T0{t_num}"
                local_slope = round(res_meta["mean_slope_deg"] + (t_num - 3) * 0.6, 2)
                # Физическая модель отступания берега Delta Y (м): отрицательное = размыв (эрозия), положительное = намыв
                erosion_m = -(
                    0.38 * (ep["hs_m"] ** 1.35) * (1.0 + 0.75 * ep["surge_m"]) * res_meta["resistance_factor"]
                ) + (0.45 if ep["season"] == "summer" else -0.15) + (t_num - 3) * 0.12

                transect_rows.append((
                    sample_id, site_info["site_id"], site_code, transect_id,
                    ep["date"], ep["season"], ep["split"], res_meta["morpho"],
                    ep["wind_ms"], ep["hs_m"], ep["surge_m"], wave_energy,
                    local_slope, res_meta["resistance_factor"],
                    round(float(np.mean(ndwi)), 4), round(float(np.mean(vv_filt)), 2),
                    round(float(erosion_m), 2)
                ))

    # 1. Записываем реестр silver_manifest.json
    silver_manifest_path = os.path.join(SILVER_DIR, "silver_manifest.json")
    with open(silver_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "layer": "SILVER",
            "total_samples": len(silver_catalog),
            "train_samples": sum(1 for x in silver_catalog if x["split"] == "train"),
            "test_samples": sum(1 for x in silver_catalog if x["split"] == "test"),
            "tensor_channels": ["B2", "B3", "B4", "B8", "B11", "NDWI", "MNDWI", "SAR_VV", "SAR_VH", "DEM"],
            "samples": silver_catalog
        }, f, indent=2, ensure_ascii=False)

    # 2. Создаем и наполняем аналитическое хранилище DuckDB Feature Store
    if os.path.exists(DUCKDB_PATH):
        os.remove(DUCKDB_PATH)
    con = duckdb.connect(DUCKDB_PATH)
    con.execute("""
        CREATE TABLE coastal_feature_store (
            sample_id VARCHAR,
            site_id INTEGER,
            site_code VARCHAR,
            transect_id VARCHAR,
            observation_date DATE,
            season VARCHAR,
            dataset_split VARCHAR,
            morpho_type VARCHAR,
            wind_speed_ms DOUBLE,
            sig_wave_height_m DOUBLE,
            storm_surge_m DOUBLE,
            wave_energy_kj DOUBLE,
            slope_deg DOUBLE,
            resistance_factor DOUBLE,
            mean_ndwi DOUBLE,
            mean_sar_vv_db DOUBLE,
            shoreline_shift_m DOUBLE
        )
    """)
    con.executemany(
        "INSERT INTO coastal_feature_store VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        transect_rows
    )
    con.close()

    # 3. Рисуем визуальное доказательство слоя Silver и нормализации
    visual_out = os.path.join(SILVER_DIR, "visual_proof_silver_dataset.png")
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), dpi=160)
    fig.suptitle(
        "Слой Silver и DuckDB Feature Store: Стандартизация 10D-тензоров (NDWI, MNDWI, фильтрация SAR) и маски берега\n"
        "(Обучающая выборка TRAIN: 36 сцен / 180 профилей | Тестовая выборка TEST: 12 сцен / 60 профилей)",
        fontsize=13, fontweight="bold", y=0.98
    )

    for row_idx, (code, vis) in enumerate(visual_samples.items()):
        #Столбец 1: Водный индекс NDWI
        im0 = axes[row_idx, 0].imshow(vis["ndwi"], cmap="RdYlBu", vmin=-0.6, vmax=0.6)
        axes[row_idx, 0].set_title(f"{vis['name']}\nИндекс воды NDWI (B3-B8)/(B3+B8)", fontsize=9.5, fontweight="bold")
        axes[row_idx, 0].axis("off")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Столбец 2: Сглаженный радар Sentinel-1 (фильтр спекл-шума)
        im1 = axes[row_idx, 1].imshow(vis["vv_filt"], cmap="viridis", vmin=-22, vmax=-4)
        axes[row_idx, 1].set_title("Сглаженный SAR VV (фильтр 3x3, дБ)", fontsize=9.5)
        axes[row_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Столбец 3: Эталонная маска суша/вода и линия уреза воды (красный контур)
        axes[row_idx, 2].imshow(vis["mask"], cmap="Blues_r", vmin=0, vmax=1)
        axes[row_idx, 2].contour(vis["mask"], levels=[0.5], colors="#dc2626", linewidths=2.0)
        axes[row_idx, 2].set_title("Эталонная маска U-Net + Урез воды L(t)", fontsize=9.5, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Столбец 4: Гистограмма нормализованных каналов 10D-тензора [0..1]
        ax3 = axes[row_idx, 3]
        t10 = vis["tensor_sample"]
        ax3.hist(t10[3].ravel(), bins=35, alpha=0.65, color="#16a34a", label="NIR (B8 norm)", density=True)
        ax3.hist(t10[5].ravel(), bins=35, alpha=0.65, color="#0284c7", label="NDWI (norm)", density=True)
        ax3.hist(t10[7].ravel(), bins=35, alpha=0.65, color="#9333ea", label="SAR VV (norm)", density=True)
        ax3.set_title("Распределение признаков в 10D-тензоре [0..1]", fontsize=9)
        ax3.set_xlim(0, 1)
        ax3.grid(True, linestyle=":", alpha=0.6)
        if row_idx == 0:
            ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(visual_out)
    plt.close(fig)

    print(f"[SILVER OK] Сформировано 48 10D-тензоров (TRAIN: 36, TEST: 12) в {SILVER_DIR}")
    print(f"[DUCKDB OK] Записано {len(transect_rows)} строк в хранилище признаков: {DUCKDB_PATH}")
    print(f"[VISUAL PROOF] Визуальная панель слоя Silver сохранена: {visual_out}")

if __name__ == "__main__":
    main()
