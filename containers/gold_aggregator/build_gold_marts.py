import os
import json
import duckdb
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/gold") else os.path.abspath("data_lake")
DUCKDB_PATH = os.path.join(BASE_DIR, "feature_store", "coastal_features.duckdb")
GOLD_DIR = os.path.join(BASE_DIR, "gold")
GOLD_REPORTS_DIR = os.path.join(GOLD_DIR, "reports")

OUT_PNG = os.path.join(GOLD_REPORTS_DIR, "visual_proof_gold_aggregator_marts.png")
OUT_JSON = os.path.join(GOLD_DIR, "gold_marts_summary.json")

PG_CONN_PARAMS = {
    "host": os.environ.get("PG_HOST", "postgis-primary"),
    "port": int(os.environ.get("PG_PORT", 5432)),
    "user": os.environ.get("PG_USER", "vkr_admin"),
    "password": os.environ.get("PG_PASSWORD", "vkr_secure_pass"),
    "dbname": os.environ.get("PG_DB", "coastal_lakehouse"),
}

DRIVER_RU = {
    "wave_energy_kj": "Энергия волн Ew",
    "sig_wave_height_m": "Высота волн Hs",
    "storm_surge_m": "Штормовой нагон Δη",
    "resistance_factor": "Податливость породы K_res",
    "wind_speed_ms": "Скорость ветра V10",
    "morpho_type": "Морфотип берега",
    "slope_deg": "Уклон склона DEM",
}


def assign_risk_grade(cri: float) -> str:
    if cri >= 72.0:
        return "CRITICAL"
    elif cri >= 52.0:
        return "HIGH"
    elif cri >= 30.0:
        return "MODERATE"
    return "LOW"


def main():
    os.makedirs(GOLD_REPORTS_DIR, exist_ok=True)

    conn = psycopg2.connect(**PG_CONN_PARAMS)
    cur = conn.cursor()

    # 1. Создаем таблицу аналитической витрины gold.transect_risk_summary в PostGIS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gold.transect_risk_summary (
            transect_id VARCHAR(32) PRIMARY KEY REFERENCES silver.coastal_transects(transect_id) ON DELETE CASCADE,
            site_id INTEGER NOT NULL REFERENCES silver.monitoring_sites(site_id) ON DELETE CASCADE,
            site_code VARCHAR(32) NOT NULL,
            morpho_type VARCHAR(64) NOT NULL,
            mean_shift_m REAL NOT NULL,
            max_erosion_m REAL NOT NULL,
            cumulative_shift_m REAL NOT NULL,
            unet_mean_dice REAL NOT NULL,
            critical_events_cnt SMALLINT NOT NULL,
            cri_index REAL NOT NULL CHECK (cri_index BETWEEN 0 AND 100),
            risk_grade VARCHAR(16) NOT NULL CHECK (risk_grade IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')),
            dominant_shap_driver VARCHAR(64) NOT NULL,
            dominant_shap_impact_m REAL NOT NULL,
            normal_geom GEOMETRY(LineString, 32634) NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_gold_risk_summary_geom
        ON gold.transect_risk_summary USING GIST (normal_geom);
        TRUNCATE TABLE gold.transect_risk_summary CASCADE;
    """)

    # 2. Выгружаем все 240 прогнозов, геометрии трансектов и метрики U-Net для агрегации
    cur.execute("""
        SELECT 
            ef.transect_id,
            ct.site_id,
            ms.site_code,
            ms.site_name,
            ms.morpho_type,
            ms.resistance_coeff,
            ef.forecast_date::text AS f_date,
            ef.predicted_shift_m,
            ef.actual_shift_m,
            ef.risk_category,
            ef.shap_values,
            sv.dice_score,
            ST_AsText(ct.normal_geom) AS wkt_geom,
            ST_AsGeoJSON(ct.normal_geom) AS geojson_geom
        FROM gold.erosion_forecasts ef
        JOIN silver.coastal_transects ct ON ct.transect_id = ef.transect_id
        JOIN silver.monitoring_sites ms ON ms.site_id = ct.site_id
        JOIN gold.shoreline_vectors sv ON sv.sample_id = ef.sample_id
        ORDER BY ct.site_id, ct.transect_num, ef.forecast_date;
    """)
    rows = cur.fetchall()

    cols = [
        "transect_id", "site_id", "site_code", "site_name", "morpho_type",
        "resistance_coeff", "f_date", "predicted_shift_m", "actual_shift_m",
        "risk_category", "shap_values", "dice_score", "wkt_geom", "geojson_geom"
    ]
    df = pd.DataFrame(rows, columns=cols)

    # 3. Агрегация показателей по каждому из 20 береговых трансектов
    mart_records_pg = []
    mart_records_json = []

    for tid, grp in df.groupby("transect_id", sort=False):
        first = grp.iloc[0]
        site_id = int(first["site_id"])
        site_code = str(first["site_code"])
        morpho = str(first["morpho_type"])
        res_coeff = float(first["resistance_coeff"])

        preds = grp["predicted_shift_m"].to_numpy(dtype=float)
        mean_shift = round(float(np.mean(preds)), 3)
        max_erosion = round(float(np.min(preds)), 3)  # самое глубокое отступание (<0)
        cum_shift = round(float(np.sum(preds)), 2)
        mean_dice = round(float(grp["dice_score"].mean()), 4)
        crit_cnt = int(np.sum(preds <= -3.5))
        crit_prob = crit_cnt / float(len(preds))

        # Агрегируем вклады Шепли по всем штормовым эпохам данного трансекта
        shap_accum = {}
        for sv_raw in grp["shap_values"]:
            sv_dict = sv_raw if isinstance(sv_raw, dict) else json.loads(sv_raw)
            for k, v in sv_dict["contributions_m"].items():
                shap_accum.setdefault(k, []).append(abs(float(v)))

        mean_shap_by_feat = {k: float(np.mean(v)) for k, v in shap_accum.items()}
        dom_feat, dom_val = max(mean_shap_by_feat.items(), key=lambda x: x[1])
        dom_val = round(dom_val, 3)

        # Вычисляем интегральный индекс риска CRI [0..100]
        c_max = min(abs(max_erosion) / 6.5, 1.0)
        c_mean = min(abs(mean_shift) / 3.2, 1.0)
        c_res = min(res_coeff / 1.5, 1.0)
        cri_val = round(float(np.clip(100.0 * (0.40 * c_max + 0.25 * c_mean + 0.20 * c_res + 0.15 * crit_prob), 0.0, 100.0)), 2)
        grade = assign_risk_grade(cri_val)

        mart_records_pg.append((
            tid, site_id, site_code, morpho,
            mean_shift, max_erosion, cum_shift, mean_dice,
            crit_cnt, cri_val, grade, dom_feat, dom_val, first["wkt_geom"]
        ))

        mart_records_json.append({
            "transect_id": tid,
            "site_id": site_id,
            "site_code": site_code,
            "site_name": str(first["site_name"]),
            "morpho_type": morpho,
            "mean_shift_m": mean_shift,
            "max_erosion_m": max_erosion,
            "cumulative_shift_m": cum_shift,
            "unet_mean_dice": mean_dice,
            "critical_events_cnt": crit_cnt,
            "cri_index": cri_val,
            "risk_grade": grade,
            "dominant_shap_driver": dom_feat,
            "dominant_shap_driver_ru": DRIVER_RU.get(dom_feat, dom_feat),
            "dominant_shap_impact_m": dom_val,
            "geometry": json.loads(first["geojson_geom"]),
        })

    execute_values(cur, """
        INSERT INTO gold.transect_risk_summary (
            transect_id, site_id, site_code, morpho_type,
            mean_shift_m, max_erosion_m, cumulative_shift_m, unet_mean_dice,
            critical_events_cnt, cri_index, risk_grade,
            dominant_shap_driver, dominant_shap_impact_m, normal_geom
        ) VALUES %s;
    """, mart_records_pg, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 32634))")

    conn.commit()
    cur.close()
    conn.close()

    # 4. Публикация аналитической витрины в колоночную СУБД DuckDB
    df_mart = pd.DataFrame(mart_records_json).drop(columns=["geometry"])
    con_duck = duckdb.connect(DUCKDB_PATH, read_only=False)
    con_duck.execute("CREATE OR REPLACE TABLE gold_transect_risk_mart AS SELECT * FROM df_mart;")
    con_duck.close()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"total_transects": len(mart_records_json), "marts": mart_records_json}, f, indent=2, ensure_ascii=False)

    # ==========================================================================
    # 5. ГЕНЕРАЦИЯ НАУЧНОГО ДАШБОРДА АГРЕГАЦИИ ВИТРИН GOLD (4 ПАНЕЛИ ДЛЯ ВКР)
    # ==========================================================================
    fig = plt.figure(figsize=(19, 13), dpi=175)
    fig.suptitle(
        "Слой Gold (Контейнер 5 — gold-aggregator): Пространственно-временная агрегация динамики берега и витрина индексов риска CRI\n"
        "Синхронная публикация аналитических витрин в СУБД PostGIS (gold.transect_risk_summary) и DuckDB (gold_transect_risk_mart) по 20 трансектам",
        fontsize=12.5, fontweight="bold", y=0.975
    )

    # Панель 1 (Верхняя левая): Пространственно-временная тепловая карта (20 трансектов x 12 эпох)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    piv = df.pivot(index="transect_id", columns="f_date", values="predicted_shift_m")
    ordered_tids = [r["transect_id"] for r in mart_records_json]
    piv = piv.loc[ordered_tids]
    short_tids = [
        t.replace("curonian_root_", "Курш.")
         .replace("svetlogorsk_cliff_", "Светл.")
         .replace("yantarny_beach_", "Янт.")
         .replace("baltiysk_spit_", "Балт.")
        for t in ordered_tids
    ]
    im1 = ax1.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlBu", vmin=-6.2, vmax=0.8)
    ax1.set_yticks(range(len(short_tids)))
    ax1.set_yticklabels(short_tids, fontsize=8, fontweight="bold")
    ax1.set_xticks(range(len(piv.columns)))
    ax1.set_xticklabels(list(piv.columns), rotation=40, ha="right", fontsize=8)
    ax1.set_title("1. Пространственно-временная матрица динамики берега Δy_hat (20 трансектов × 12 эпох)", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Смещение уреза (м)")

    # Панель 2 (Верхняя правая): Рейтинг трансектов по интегральному индексу геоэкологического риска CRI [0..100]
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    df_sorted_cri = df_mart.sort_values("cri_index", ascending=True)
    y_lbls = [
        t.replace("curonian_root_", "Курш. ")
         .replace("svetlogorsk_cliff_", "Светл. ")
         .replace("yantarny_beach_", "Янт. ")
         .replace("baltiysk_spit_", "Балт. ")
        for t in df_sorted_cri["transect_id"]
    ]
    cri_vals = df_sorted_cri["cri_index"].to_numpy()
    grade_colors = {
        "CRITICAL": "#dc2626",
        "HIGH": "#ea580c",
        "MODERATE": "#0284c7",
        "LOW": "#16a34a"
    }
    bar_cols = [grade_colors[g] for g in df_sorted_cri["risk_grade"]]
    bars = ax2.barh(y_lbls, cri_vals, color=bar_cols, edgecolor="black", height=0.68)
    for b, v, g in zip(bars, cri_vals, df_sorted_cri["risk_grade"]):
        ax2.text(v + 1.0, b.get_y() + b.get_height()/2, f"{v:.1f} ({g})", va="center", ha="left", fontsize=7.8, fontweight="bold")
    ax2.axvline(72.0, color="#991b1b", linestyle="--", linewidth=1.4, label="Порог CRITICAL (CRI >= 72)")
    ax2.axvline(52.0, color="#ea580c", linestyle=":", linewidth=1.4, label="Порог HIGH (CRI >= 52)")
    ax2.set_xlim(0, 108)
    ax2.set_title("2. Ранжирование 20 береговых профилей по интегральному индексу риска CRI [0..100]", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Интегральный индекс геоэкологического риска CRI (баллы 0..100)")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8)

    # Панель 3 (Нижняя левая): Кумулятивная деградация берега по 4 полигонам за 2022-2025 гг.
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    site_palette = {
        "curonian_root": ("#dc2626", "o-", "Корень Куршской косы (Зеленоградск)"),
        "baltiysk_spit": ("#9333ea", "s-", "Балтийская коса и пролив"),
        "svetlogorsk_cliff": ("#2563eb", "^--", "Абразионный клиф Светлогорска"),
        "yantarny_beach": ("#16a34a", "d-.", "Широкий морской пляж Янтарного"),
    }
    for scode, (col, style, sname) in site_palette.items():
        sub_s = df[df["site_code"] == scode].groupby("f_date")["predicted_shift_m"].mean()
        cum_s = np.cumsum(sub_s.to_numpy())
        ax3.plot(list(sub_s.index), cum_s, style, color=col, linewidth=2.3, markersize=6, label=f"{sname} (итог: {cum_s[-1]:.1f} м)")
    ax3.set_xticklabels(list(piv.columns), rotation=35, ha="right", fontsize=8.2)
    ax3.set_title("3. Кумулятивное отступание береговой линии ∑Δy(t) по полигонам (2022–2025 гг.)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Накопленное смещение уреза воды с 2022 г. (м)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=8.3)

    # Панель 4 (Нижняя правая): Сводная витрина по 4 полигонам (выборка из gold.transect_risk_summary)
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax4.axis("off")
    ax4.set_title("4. Агрегированная сводка витрины gold.transect_risk_summary по участкам мониторинга", fontsize=10, fontweight="bold")

    site_agg = df_mart.groupby(["site_code", "morpho_type"], sort=False).agg(
        mean_cri=("cri_index", "mean"),
        max_cri=("cri_index", "max"),
        max_eros=("max_erosion_m", "min"),
        cum_eros=("cumulative_shift_m", "mean"),
        mean_dice=("unet_mean_dice", "mean"),
        dom_driver=("dominant_shap_driver_ru", lambda s: s.mode().iloc[0])
    ).reset_index()

    col_headers = ["Участок", "Морфотип", "Ср. CRI", "Макс. CRI", "Пик шторма", "Сумма за 4 г.", "U-Net Dice", "Главный фактор SHAP"]
    tbl_rows = [
        [
            r["site_code"], r["morpho_type"],
            f"{r['mean_cri']:.1f}", f"{r['max_cri']:.1f}",
            f"{r['max_eros']:.2f} м", f"{r['cum_eros']:.1f} м",
            f"{r['mean_dice']:.4f}", r["dom_driver"]
        ]
        for _, r in site_agg.iterrows()
    ]
    tbl = ax4.table(cellText=tbl_rows, colLabels=col_headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.3)
    tbl.scale(1.0, 2.35)
    for (r_i, c_i), cell in tbl.get_celld().items():
        if r_i == 0:
            cell.set_facecolor("#065f46")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_i % 2 == 0:
            cell.set_facecolor("#ecfdf5")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PNG)
    plt.close(fig)

    print(f"[CONTAINER 5 OK] Витрина gold.transect_risk_summary (20 профилей) записана в PostGIS и DuckDB!")
    print(f"[VISUAL PROOF] Дашборд витрин слоя Gold сохранен: {OUT_PNG}")


if __name__ == "__main__":
    main()
