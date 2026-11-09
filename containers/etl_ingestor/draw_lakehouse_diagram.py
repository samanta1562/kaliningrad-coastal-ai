import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake") else os.path.abspath("data_lake")
OUT_PNG = os.path.join(BASE_DIR, "visual_architecture_lakehouse_lineage.png")

def draw_box(ax, x, y, w, h, title, items, header_color, bg_color, border_color="#334155"):
    # Основной блок
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.8
    )
    ax.add_patch(box)
    # Шапка блока
    hdr_h = 0.048
    hdr = FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h, boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=header_color, edgecolor=border_color, linewidth=1.5
    )
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - hdr_h / 2, title, color="white", fontsize=9.5,
            fontweight="bold", ha="center", va="center")
    # Текст внутри блока
    body_text = "\n".join(items)
    ax.text(x + 0.015, y + h - hdr_h - 0.02, body_text, color="#0f172a",
            fontsize=8.2, ha="left", va="top", family="monospace", linespacing=1.32)

def draw_arrow(ax, x1, y1, x2, y2, label="", color="#1e293b"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>,head_length=7,head_width=4.5",
        color=color, linewidth=1.8, connectionstyle="arc3,rad=0.0"
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.012, label, fontsize=7.8, fontweight="bold",
                color=color, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 11.5), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Архитектура геопространственного хранилища Data Lakehouse и движение данных между слоями (Data Lineage)\n"
        "Файловый том Docker (coastal_data_lake_vol)  +  DuckDB Feature Store (OLAP)  +  СУБД PostGIS 3.4 (3НФ / GiST / GIN)",
        fontsize=13, fontweight="bold", y=0.97, color="#0f172a"
    )

    # Заголовки колонок (Слои Bronze -> Silver -> Gold)
    col_headers = [
        (0.02, 0.28, "СЛОЙ 1: BRONZE (Сырые данные и аудит)", "#92400e", "#fef3c7"),
        (0.35, 0.31, "СЛОЙ 2: SILVER (Нормализация и 3НФ)", "#1e40af", "#dbeafe"),
        (0.70, 0.28, "СЛОЙ 3: GOLD (Витрина ИИ и ГИС)", "#065f46", "#d1fae5"),
    ]
    for cx, cw, ctitle, c_edge, c_bg in col_headers:
        bg_col = FancyBboxPatch(
            (cx, 0.03), cw, 0.88, boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=c_bg, edgecolor=c_edge, linewidth=1.5, linestyle="--", alpha=0.35
        )
        ax.add_patch(bg_col)
        ax.text(cx + cw / 2, 0.895, ctitle, color=c_edge, fontsize=10.5,
                fontweight="bold", ha="center", va="center")

    # --- БЛОКИ СЛОЯ BRONZE ---
    draw_box(
        ax, 0.035, 0.56, 0.25, 0.29,
        "Файловый том: /data_lake/bronze/",
        [
            "• {site}_S2_L2A.tif (5 каналов)",
            "• {site}_S1_SAR.tif (VV, VH дБ)",
            "• {site}_COP_DEM.tif (рельеф, м)",
            "• {site}_ERA5_meteo.json (72 ч)",
            "• manifest.json (паспорт + SHA256)",
            "----------------------------------",
            "Проекция: EPSG:32634 (UTM 34N)",
            "Фильтр NoData: < 2% (без пустот)"
        ],
        header_color="#b45309", bg_color="#fffbeb"
    )

    draw_box(
        ax, 0.035, 0.10, 0.25, 0.36,
        "PostGIS [SCHEMA bronze]",
        [
            "[Таблица: bronze.raw_ingestion_log]",
            "• PK: ingestion_id (BIGSERIAL)",
            "• site_code (VARCHAR)",
            "• source_type (S2 / S1 / DEM / ERA5)",
            "• scene_or_batch_id (VARCHAR)",
            "• acquisition_date (DATE)",
            "• file_path (TEXT -> Shared Vol)",
            "• sha256_checksum (UNIQUE CHAR(64))",
            "• raw_metadata (JSONB, Индекс GIN)",
            "• ingested_at (TIMESTAMPTZ)"
        ],
        header_color="#92400e", bg_color="#fef3c7"
    )

    # --- БЛОКИ СЛОЯ SILVER ---
    draw_box(
        ax, 0.365, 0.62, 0.28, 0.23,
        "Файловый том: /data_lake/silver/",
        [
            "• /train/*_tensor10d.npy (36 шт.)",
            "• /test/*_tensor10d.npy  (12 шт.)",
            "• *_mask.npy (эталон суша/вода)",
            "------------------------------------",
            "Форма тензора: (10, 256, 256)",
            "Каналы: B2,B3,B4,B8,B11,NDWI,MNDWI,",
            "        SAR_VV, SAR_VH, DEM [0..1]"
        ],
        header_color="#1d4ed8", bg_color="#eff6ff"
    )

    draw_box(
        ax, 0.365, 0.41, 0.28, 0.17,
        "DuckDB Feature Store (OLAP)",
        [
            "[Таблица: coastal_feature_store]",
            "• 240 строк (180 train / 60 test)",
            "• 20 трансектов x 12 штормовых эпох",
            "• Признаки: wind_ms, hs_m, surge_m,",
            "  wave_energy_kj, slope_deg, ndwi",
            "• Целевая: shoreline_shift_m (м)"
        ],
        header_color="#0369a1", bg_color="#e0f2fe"
    )

    draw_box(
        ax, 0.365, 0.05, 0.28, 0.32,
        "PostGIS [SCHEMA silver] (Ядро 3НФ)",
        [
            "1. silver.monitoring_sites",
            "   PK: site_id | bbox_geom [GiST]",
            "2. silver.coastal_transects (20 шт.)",
            "   PK: transect_id | FK: site_id",
            "   normal_geom LineString [GiST]",
            "3. silver.era5_timeseries_clean",
            "   PK: obs_id | FK: site_id [B-Tree]",
            "4. silver.ml_samples_registry",
            "   PK: sample_id | FK: ingestion_id"
        ],
        header_color="#1e3a8a", bg_color="#dbeafe"
    )

    # --- БЛОКИ СЛОЯ GOLD ---
    draw_box(
        ax, 0.715, 0.56, 0.25, 0.29,
        "Модели ИИ: /data_lake/gold/",
        [
            "• Attention U-Net (PyTorch .pt)",
            "  Сегментация уреза воды L(t)",
            "  Метрики: Dice >= 0.91, IoU >= 0.85",
            "----------------------------------",
            "• CatBoost + TreeSHAP (.cbm)",
            "  Прогноз размыва Delta Y (м)",
            "  Метрики: MAE <= 0.85 м, R2 >= 0.84",
            "  Объяснимость: вклады Шепли"
        ],
        header_color="#047857", bg_color="#ecfdf5"
    )

    draw_box(
        ax, 0.715, 0.10, 0.25, 0.40,
        "PostGIS [SCHEMA gold] + CQRS",
        [
            "[1. gold.shoreline_vectors]",
            "• PK: shoreline_id | FK: sample_id",
            "• dice_score, iou_score (REAL)",
            "• shoreline_geom (MultiLineString,",
            "  проекция EPSG:32634, индекс GiST)",
            "----------------------------------",
            "[2. gold.erosion_forecasts]",
            "• PK: forecast_id | FK: transect_id",
            "• predicted_shift_m, risk_category",
            "• shap_values (JSONB, индекс GIN)",
            "-> Репликация WAL на postgis-replica"
        ],
        header_color="#065f46", bg_color="#d1fae5"
    )

    # --- СТРЕЛКИ ДВИЖЕНИЯ ДАННЫХ (DATA LINEAGE) ---
    draw_arrow(ax, 0.16, 0.56, 0.16, 0.46, "SHA-256 + JSONB", "#92400e")
    draw_arrow(ax, 0.285, 0.72, 0.365, 0.72, "Z-score / Min-Max\nNDWI, Фильтр Ли", "#1d4ed8")
    draw_arrow(ax, 0.285, 0.62, 0.365, 0.50, "Расчет энергии волн\nпо 20 трансектам", "#0369a1")
    draw_arrow(ax, 0.285, 0.25, 0.365, 0.21, "FK: ingestion_id\nНормализация 3НФ", "#1e3a8a")
    draw_arrow(ax, 0.645, 0.72, 0.715, 0.72, "Тензоры 10D\n(Train 36 / Test 12)", "#047857")
    draw_arrow(ax, 0.645, 0.50, 0.715, 0.63, "Признаки OLAP\n(Train 180 / Test 60)", "#047857")
    draw_arrow(ax, 0.84, 0.56, 0.84, 0.50, "Инференс и\nвекторизация", "#065f46")
    draw_arrow(ax, 0.645, 0.21, 0.715, 0.25, "FK: site_id,\ntransect_id, sample_id", "#065f46")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_PNG)
    plt.close(fig)
    print(f"[DIAGRAM OK] Архитектурная схема успешно сохранена: {OUT_PNG}")

if __name__ == "__main__":
    main()