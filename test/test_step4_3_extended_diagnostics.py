import os
import json

BASE_DIR = "/data_lake" if os.path.isdir("/data_lake/gold") else "data_lake"
GOLD_MODELS_DIR = os.path.join(BASE_DIR, "gold", "models")
GOLD_REPORTS_DIR = os.path.join(BASE_DIR, "gold", "reports")

def test_extended_diagnostics_artifacts_and_unbiased_residuals():
    """Проверка генерации полотна из 4 графиков, несмещенности остатков CatBoost и превосходства 10D-тензора в Ablation Study."""
    png_path = os.path.join(GOLD_REPORTS_DIR, "visual_proof_gold_extended_diagnostics.png")
    json_path = os.path.join(GOLD_MODELS_DIR, "extended_diagnostics_metrics.json")

    assert os.path.isfile(png_path) and os.path.getsize(png_path) > 120_000, "График расширенной диагностики не создан"
    assert os.path.isfile(json_path), "JSON-паспорт расширенной диагностики не найден"

    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # 1. Проверка отсутствия систематического смещения остатков (|mean(e)| < 0.25 м)
    assert abs(d["residuals_test_mean_m"]) < 0.25, f"Обнаружено систематическое смещение прогноза: {d['residuals_test_mean_m']} м"

    # 2. Проверка доказательства мультимодальности (Ablation Study): полный 10D-тензор превосходит урезанные конфигурации
    ab = d["unet_ablation_iou"]
    assert ab["Full_10D_S2_SAR_DEM"] > ab["Without_DEM_9D"]
    assert ab["Without_DEM_9D"] > ab["Without_SAR_8D_OpticOnly"]
    assert ab["Without_SAR_8D_OpticOnly"] > ab["RGB_Only_3D_Baseline"]

    # 3. Проверка попиксельной точности классификации (Overall Pixel Accuracy >= 95%)
    cm = d["unet_pixel_confusion_matrix"]
    assert cm["overall_pixel_accuracy"] >= 0.95


