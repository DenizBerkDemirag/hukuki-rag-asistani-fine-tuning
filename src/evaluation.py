"""
Model değerlendirme: Test veri kümesi üzerinde metrikler hesaplama.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.data_utils import load_json


def evaluate_model(model, tokenizer, test_data, max_samples=50):
    """
    Test verisinden örnekler üzerinde model cevaplarını üretir
    ve basit karşılaştırma metrikleri döner.
    """
    from src.inference import generate_answer

    results = []
    for i, item in enumerate(test_data[:max_samples]):
        soru = item.get("Soru", "").strip()
        beklenen = item.get("Cevap", "").strip()
        if not soru or not beklenen:
            continue

        uretilen = generate_answer(model, tokenizer, soru)

        # Basit kelime örtüşme skoru
        beklenen_kelimeler = set(beklenen.lower().split())
        uretilen_kelimeler = set(uretilen.lower().split())
        if beklenen_kelimeler:
            ortusme = len(beklenen_kelimeler & uretilen_kelimeler) / len(beklenen_kelimeler)
        else:
            ortusme = 0.0

        results.append({
            "soru": soru,
            "beklenen": beklenen,
            "uretilen": uretilen,
            "kelime_ortusme": round(ortusme, 3),
        })

    if not results:
        return {"ortalama_ortusme": 0.0, "ornekler": []}

    ort = sum(r["kelime_ortusme"] for r in results) / len(results)
    return {
        "ortalama_ortusme": round(ort, 3),
        "degerlendirilen_sayisi": len(results),
        "ornekler": results,
    }


def compute_dataset_stats(file_path: str) -> dict:
    """Veri kümesinin temel istatistiklerini hesaplar (model gerekmez)."""
    data = load_json(file_path)

    soru_uzunluklari = []
    cevap_uzunluklari = []

    for item in data:
        soru = item.get("Soru", "")
        cevap = item.get("Cevap", "")
        if soru:
            soru_uzunluklari.append(len(soru.split()))
        if cevap:
            cevap_uzunluklari.append(len(cevap.split()))

    return {
        "toplam_kayit": len(data),
        "ort_soru_kelime": round(sum(soru_uzunluklari) / max(len(soru_uzunluklari), 1), 1),
        "ort_cevap_kelime": round(sum(cevap_uzunluklari) / max(len(cevap_uzunluklari), 1), 1),
        "max_soru_kelime": max(soru_uzunluklari, default=0),
        "max_cevap_kelime": max(cevap_uzunluklari, default=0),
        "min_soru_kelime": min(soru_uzunluklari, default=0),
        "min_cevap_kelime": min(cevap_uzunluklari, default=0),
    }
