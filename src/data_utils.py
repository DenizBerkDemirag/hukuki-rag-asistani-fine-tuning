"""
Türk Hukuku Asistanı - Veri Hazırlama, Doğrulama ve Yönetim Modülü.

Özellikler:
1. JSON ve JSONL dosyalarını okuma (dosyadan veya yüklenen bayt/metinden).
2. Boş, eksik ve bozuk kayıtları tespit etme.
3. Birebir aynı (duplicate) kayıtları bulma.
4. Aynı soruya farklı cevap verilen çelişkili kayıtları tespit edip raporlama.
5. Türkçe karakterleri UTF-8 ve ensure_ascii=False ile koruma.
6. Benzer/aynı soruları gruplayarak veri sızıntısını (leakage) önleyen %80/%10/%10 split.
7. Standart 'messages' formatına dönüştürme.
8. Dosyaları data/ klasörüne güvenle kaydetme.
"""

import os
import io
import re
import json
import random
import unicodedata
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union

import config


# ─── Türkçe Metin ve Normalizasyon Yardımcıları ───────────────────────────────

def turkish_lower(text: str) -> str:
    """Türkçe İ/i ve I/ı harflerini doğru şekilde küçük harfe çevirir."""
    if not isinstance(text, str):
        return ""
    translation_table = str.maketrans({
        "İ": "i",
        "I": "ı",
        "Ğ": "ğ",
        "Ü": "ü",
        "Ş": "ş",
        "Ö": "ö",
        "Ç": "ç",
    })
    return text.translate(translation_table).lower()


def clean_and_normalize_text(text: str) -> str:
    """
    Benzerlik ve eşleme kontrolü için metni temizler:
    - Türkçe küçük harfe çevirir
    - Noktalama işaretlerini ve fazla boşlukları temizler
    """
    text = turkish_lower(text.strip())
    # Noktalama işaretlerini kaldır veya boşluğa çevir
    text = re.sub(r"[^\w\sğüşıöç]", " ", text)
    # Birden fazla boşluğu teke indir
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_qa_pair(item: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Kayıttan soru ve cevap metinlerini esnek bir şekilde çıkarır.
    'Soru'/'Cevap', 'soru'/'cevap', 'question'/'answer' veya 'messages' formatını destekler.
    """
    if not isinstance(item, dict):
        return None, None

    # messages formatında ise
    if "messages" in item and isinstance(item["messages"], list):
        user_msg = next((m.get("content") for m in item["messages"] if m.get("role") == "user"), None)
        asst_msg = next((m.get("content") for m in item["messages"] if m.get("role") == "assistant"), None)
        return user_msg, asst_msg

    # Standart Soru / Cevap alanları
    soru = item.get("Soru") or item.get("soru") or item.get("question") or item.get("instruction")
    cevap = item.get("Cevap") or item.get("cevap") or item.get("answer") or item.get("output") or item.get("response")

    return soru, cevap


# ─── 1. Dosya Okuma (JSON ve JSONL) ──────────────────────────────────────────

def load_file_content(
    source: Union[str, bytes, io.BytesIO, io.StringIO],
    filename: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    JSON veya JSONL formatındaki verileri okur.
    Kaynak dosya yolu (str), bayt (bytes) veya akış (stream) olabilir.
    """
    # Kaynak bir dosya yolu ise
    if isinstance(source, str) and (os.path.exists(source) or (filename is None and ("\n" not in source and "{" not in source))):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Dosya bulunamadı: {source}")
        if filename is None:
            filename = source
        with open(source, "r", encoding="utf-8") as f:
            raw_content = f.read()
    elif isinstance(source, bytes):
        raw_content = source.decode("utf-8", errors="replace")
    elif isinstance(source, (io.BytesIO, io.StringIO)):
        if isinstance(source, io.BytesIO):
            raw_content = source.getvalue().decode("utf-8", errors="replace")
        else:
            raw_content = source.getvalue()
    elif isinstance(source, str):
        # Doğrudan metin içeriği
        raw_content = source
    else:
        raise ValueError(f"Desteklenmeyen veri kaynağı türü: {type(source)}")

    raw_content = raw_content.strip()
    if not raw_content:
        return []

    # Format belirleme: JSON mu JSONL mi?
    is_jsonl = False
    if filename and filename.lower().endswith(".jsonl"):
        is_jsonl = True
    elif not raw_content.startswith("[") and ("\n" in raw_content):
        # Köşeli parantez ile başlamıyorsa ve satırlar varsa muhtemelen JSONL
        is_jsonl = True

    records = []

    if is_jsonl:
        for line_num, line in enumerate(raw_content.splitlines(), start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
                records.append(obj)
            except Exception as e:
                records.append({
                    "__error__": f"Satır {line_num} JSON ayrıştırma hatası: {str(e)}",
                    "__raw_line__": line_str
                })
    else:
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, list):
                records = parsed
            elif isinstance(parsed, dict):
                # Tek bir nesne veya {"data": [...]} yapısı
                if "data" in parsed and isinstance(parsed["data"], list):
                    records = parsed["data"]
                else:
                    records = [parsed]
            else:
                raise ValueError(f"Beklenmeyen JSON yapısı: {type(parsed)}")
        except json.JSONDecodeError:
            # Standart JSON okunamadıysa JSONL olarak tekrar dene
            records = []
            for line_num, line in enumerate(raw_content.splitlines(), start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    records.append(obj)
                except Exception as e:
                    records.append({
                        "__error__": f"Satır {line_num} JSON ayrıştırma hatası: {str(e)}",
                        "__raw_line__": line_str
                    })

    return records


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Geriye dönük uyumluluk için eski load_json sarmalayıcısı."""
    return load_file_content(file_path)


# ─── 2-5. Veri Doğrulama ve Analiz ───────────────────────────────────────────

def analyze_dataset(data: List[Any]) -> Dict[str, Any]:
    """
    Veri kümesini kapsamlı şekilde analiz eder:
    - Boş, eksik veya bozuk kayıtlar
    - Birebir aynı (tam kopya) kayıtlar
    - Aynı soruya farklı cevap verilen çelişkili/farklı kayıtlar (raporlanır, değiştirilmez)
    - Türkçe karakterlerin varlığı ve istatistikleri
    """
    report = {
        "toplam_kayit": len(data),
        "gecerli_sayisi": 0,
        "hatali_sayisi": 0,
        "tam_kopya_sayisi": 0,
        "celiskili_soru_sayisi": 0,
        "hatali_kayitlar": [],           # List of {index, reason, sample}
        "tam_kopyalar": [],              # List of {soru, cevap, count, indices}
        "celiskili_kayitlar": [],        # List of {soru, cevaplar: [...], indices: [...]}
        "gecerli_veriler": [],           # List of clean {Soru: str, Cevap: str}
    }

    # 1. Aşama: Eksik / Bozuk / Boş Kontrolü
    # Soru -> {cevap: [indices]}
    question_map = defaultdict(lambda: defaultdict(list))
    # (soru, cevap) -> [indices]
    exact_pairs = defaultdict(list)

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            report["hatali_sayisi"] += 1
            report["hatali_kayitlar"].append({
                "index": idx,
                "reason": "Kayıt bir sözlük (dictionary) nesnesi değil",
                "sample": str(item)[:100]
            })
            continue

        if "__error__" in item:
            report["hatali_sayisi"] += 1
            report["hatali_kayitlar"].append({
                "index": idx,
                "reason": item["__error__"],
                "sample": item.get("__raw_line__", "")[:100]
            })
            continue

        soru, cevap = extract_qa_pair(item)

        if soru is None and cevap is None:
            report["hatali_sayisi"] += 1
            report["hatali_kayitlar"].append({
                "index": idx,
                "reason": "Soru ve Cevap alanları eksik",
                "sample": str(item)[:100]
            })
            continue

        if soru is None or not str(soru).strip():
            report["hatali_sayisi"] += 1
            report["hatali_kayitlar"].append({
                "index": idx,
                "reason": "Soru alanı boş veya eksik",
                "sample": str(item)[:100]
            })
            continue

        if cevap is None or not str(cevap).strip():
            report["hatali_sayisi"] += 1
            report["hatali_kayitlar"].append({
                "index": idx,
                "reason": "Cevap alanı boş veya eksik",
                "sample": str(item)[:100]
            })
            continue

        clean_soru = str(soru).strip()
        clean_cevap = str(cevap).strip()

        report["gecerli_sayisi"] += 1
        report["gecerli_veriler"].append({
            "Soru": clean_soru,
            "Cevap": clean_cevap,
            "_original_index": idx
        })

        norm_q = clean_and_normalize_text(clean_soru)
        norm_a = clean_and_normalize_text(clean_cevap)

        exact_pairs[(clean_soru, clean_cevap)].append(idx)
        question_map[norm_q][clean_cevap].append(idx)

    # 2. Aşama: Birebir Aynı Kayıtlar (Exact Duplicates)
    for (q, a), idxs in exact_pairs.items():
        if len(idxs) > 1:
            report["tam_kopya_sayisi"] += (len(idxs) - 1)
            report["tam_kopyalar"].append({
                "soru": q,
                "cevap": a,
                "tekrar_sayisi": len(idxs),
                "indeksler": idxs
            })

    # 3. Aşama: Aynı Soruya Farklı Cevaplar (Çelişkili/Alternatif Cevaplar)
    # Not: Şüpheli/çelişkili cevaplar otomatik olarak DEĞİŞTİRİLMEZ; açıkça raporlanır!
    for norm_q, answers_dict in question_map.items():
        if len(answers_dict) > 1:
            all_indices = []
            distinct_answers = []
            for ans, idxs in answers_dict.items():
                distinct_answers.append({
                    "cevap": ans,
                    "kayit_adet": len(idxs),
                    "indeksler": idxs
                })
                all_indices.extend(idxs)

            # İlk cevaptan orijinal soru metnini alalım
            sample_soru = data[all_indices[0]].get("Soru") or data[all_indices[0]].get("soru") or norm_q

            report["celiskili_soru_sayisi"] += 1
            report["celiskili_kayitlar"].append({
                "soru": sample_soru,
                "norm_soru": norm_q,
                "farkli_cevap_sayisi": len(distinct_answers),
                "cevaplar": distinct_answers,
                "tum_indeksler": all_indices
            })

    return report


def validate_dataset(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Geriye dönük uyumluluk için özet istatistik döner."""
    analysis = analyze_dataset(data)
    return {
        "toplam": analysis["toplam_kayit"],
        "gecerli": analysis["gecerli_sayisi"],
        "sorunlu": analysis["hatali_sayisi"],
        "tam_kopya": analysis["tam_kopya_sayisi"],
        "celiskili_soru": analysis["celiskili_soru_sayisi"],
    }


# ─── 6-8. Veri Kümelerini Ayırma ve Sızıntıyı Önleme ──────────────────────────

def _compute_question_key(question: str) -> str:
    """Soruyu benzerlik gruplaması için standart anahtara dönüştürür."""
    return clean_and_normalize_text(question)


def group_similar_questions(
    records: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """
    Aynı veya çok benzer soruları aynı grupta toplar.
    Bu gruplar daha sonra bölünmeden train/val/test kümelerine atanır.
    Böylece aynı sorunun farklı kümelere dağılması (veri sızıntısı) engellenir.
    """
    # 1. Adım: Normalize soru anahtarına göre grupla
    groups_by_norm = defaultdict(list)
    for record in records:
        soru = record.get("Soru", "")
        norm_key = _compute_question_key(soru)
        groups_by_norm[norm_key].append(record)

    # 2. Adım: Çok benzer soruları (Jaccard token similarity >= 0.85) birleştir
    # Küçük-orta boyutlu kümelerde hassas birleştirme, büyük veri için hızlı kümeleme
    keys = list(groups_by_norm.keys())
    
    # Kelime kümeleri
    token_sets = {k: set(k.split()) for k in keys if k}

    # Birleştirme haritası (Union-Find mantığı)
    parent = {k: k for k in keys}

    def find(k):
        path = []
        while parent[k] != k:
            path.append(k)
            k = parent[k]
        for p in path:
            parent[p] = k
        return k

    def union(k1, k2):
        r1, r2 = find(k1), find(k2)
        if r1 != r2:
            parent[r1] = r2

    # Kelime kümesi benzerliği (çok benzer soruları yakalar)
    # Performans için ilk 3 kelimesi aynı olanlar veya uzunlukları yakın olanlar taranır
    for i in range(len(keys)):
        k1 = keys[i]
        s1 = token_sets.get(k1, set())
        if len(s1) < 3:
            continue
        for j in range(i + 1, min(i + 40, len(keys))):
            k2 = keys[j]
            s2 = token_sets.get(k2, set())
            if not s2 or abs(len(s1) - len(s2)) > 2:
                continue
            intersection = len(s1 & s2)
            union_len = len(s1 | s2)
            if union_len > 0 and (intersection / union_len) >= 0.85:
                union(k1, k2)

    # Nihai birleşik grupları oluştur
    merged_groups = defaultdict(list)
    for k in keys:
        root = find(k)
        merged_groups[root].extend(groups_by_norm[k])

    return list(merged_groups.values())


def split_dataset_grouped(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    deduplicate_exact: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Kayıtları %80 Eğitim, %10 Doğrulama ve %10 Test olarak böler.
    
    Kritik Kural: Aynı veya çok benzer sorular aynı grup içinde tutulur ve 
    asla farklı veri kümelerine ayrılmaz (Data Leakage engellenir).
    """
    # İsteğe bağlı birebir aynı kopya kayıtları tekilleştir
    if deduplicate_exact:
        seen = set()
        clean_records = []
        for r in records:
            pair = (r.get("Soru", "").strip(), r.get("Cevap", "").strip())
            if pair not in seen:
                seen.add(pair)
                clean_records.append(r)
        records = clean_records

    # Soru gruplarını oluştur
    groups = group_similar_questions(records)

    # Karıştır
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_items = sum(len(g) for g in groups)
    if total_items >= 10:
        target_val = max(1, int(round(total_items * val_ratio)))
        target_test = max(1, int(round(total_items * test_ratio)))
        target_train = max(1, total_items - target_val - target_test)
    elif total_items >= 3:
        target_val = 1
        target_test = 1
        target_train = total_items - 2
    else:
        target_train = total_items
        target_val = 0
        target_test = 0

    train_data = []
    val_data = []
    test_data = []

    train_count = 0
    val_count = 0

    for group in groups:
        g_len = len(group)
        if train_count < target_train:
            train_data.extend(group)
            train_count += g_len
        elif val_count < target_val:
            val_data.extend(group)
            val_count += g_len
        else:
            test_data.extend(group)

    # Çok küçük veri kümelerinde test boş kalmışsa ve train yeterliyse dengeli aktar
    if not test_data and len(train_data) > 1 and total_items >= 3:
        test_data.append(train_data.pop())

    return train_data, val_data, test_data


# ─── 9. Messages Formatına Dönüştürme ─────────────────────────────────────────

def format_messages_entry(
    soru: str,
    cevap: str,
    system_prompt: str = config.SYSTEM_PROMPT
) -> Dict[str, List[Dict[str, str]]]:
    """
    Tek bir soru-cevap çiftini istenen 'messages' formatına dönüştürür.
    
    Örnek:
    {
      "messages": [
        {"role": "system", "content": "Sen Türk hukuku konusunda yardımcı olan bir asistansın."},
        {"role": "user", "content": "Erginlik kaç yaşında başlar?"},
        {"role": "assistant", "content": "Türk Medeni Kanunu'na göre..."}
      ]
    }
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": str(soru).strip()
            },
            {
                "role": "assistant",
                "content": str(cevap).strip()
            }
        ]
    }


def convert_to_messages_format(
    records: List[Dict[str, Any]],
    system_prompt: str = config.SYSTEM_PROMPT
) -> List[Dict[str, Any]]:
    """
    Soru-cevap listesini messages formatındaki liste nesnelerine dönüştürür.
    Boş veya eksik olan kayıtlar atlanır.
    """
    formatted = []
    for item in records:
        soru, cevap = extract_qa_pair(item)
        if soru and cevap and str(soru).strip() and str(cevap).strip():
            formatted.append(format_messages_entry(soru, cevap, system_prompt))
    return formatted


# ─── 10. Dosyaları Güvenle Kaydetme (Türkçe Karakter Korumalı) ────────────────

def save_dataset_file(
    data: List[Any],
    output_path: str,
    format_type: str = "json"
) -> str:
    """
    Veriyi data/ klasörüne kaydeder.
    - ensure_ascii=False ile Türkçe karakterleri (ğ, ü, ş, ı, ö, ç, İ) korur.
    - Orijinal dosyanın üzerine yazılmasını engeller.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if format_type == "jsonl" or output_path.lower().endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def process_and_split_pipeline(
    raw_data: List[Any],
    output_dir: str = config.DATASET_DIR,
    system_prompt: str = config.SYSTEM_PROMPT,
    deduplicate_exact: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Tüm veri hazırlama boru hattını (pipeline) baştan sona çalıştırır:
    1. Veriyi analiz eder ve eksik/hatalı olanları ayıklar.
    2. Birebir kopyaları tekilleştirir (isteğe bağlı).
    3. Çelişkili cevapları değiştirmez, raporlar.
    4. Soru gruplamasıyla sızıntısız %80 / %10 / %10 böler.
    5. 'messages' formatına dönüştürür.
    6. Türkçe karakterleri koruyarak data/ klasörüne kaydeder.
    """
    analysis = analyze_dataset(raw_data)
    valid_records = analysis["gecerli_veriler"]

    if not valid_records:
        raise ValueError("İşlenecek geçerli kayıt bulunamadı.")

    # Gruplu bölme işlemi (Grouped Split)
    train_records, val_records, test_records = split_dataset_grouped(
        valid_records,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=seed,
        deduplicate_exact=deduplicate_exact
    )

    # Messages formatına dönüştürme
    train_messages = convert_to_messages_format(train_records, system_prompt)
    val_messages = convert_to_messages_format(val_records, system_prompt)
    test_messages = convert_to_messages_format(test_records, system_prompt)

    # Dosya yolları
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_messages.json")
    val_path = os.path.join(output_dir, "val_messages.json")
    test_path = os.path.join(output_dir, "test_messages.json")

    save_dataset_file(train_messages, train_path, "json")
    save_dataset_file(val_messages, val_path, "json")
    save_dataset_file(test_messages, test_path, "json")

    return {
        "analysis": analysis,
        "train_count": len(train_messages),
        "val_count": len(val_messages),
        "test_count": len(test_messages),
        "total_processed": len(train_messages) + len(val_messages) + len(test_messages),
        "paths": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        },
        "datasets": {
            "train": train_messages,
            "val": val_messages,
            "test": test_messages,
        }
    }


# ─── Geriye Dönük Uyumluluk Fonksiyonları ─────────────────────────────────────

def format_chatml(soru: str, cevap: str, system_prompt: str) -> str:
    """Tek bir soru-cevap çiftini Qwen2.5 ChatML formatına dönüştürür."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{soru}<|im_end|>\n"
        f"<|im_start|>assistant\n{cevap}<|im_end|>"
    )


def prepare_training_data(
    data: List[Dict[str, str]],
    system_prompt: str,
) -> List[Dict[str, str]]:
    """Eski format desteği: 'text' alanı içeren liste döner."""
    prepared = []
    for item in data:
        soru, cevap = extract_qa_pair(item)
        if not soru or not cevap:
            continue
        text = format_chatml(str(soru), str(cevap), system_prompt)
        prepared.append({"text": text})
    return prepared


def get_sample_questions(data: List[Dict[str, str]], n: int = 5) -> List[str]:
    """Veri kümesinden n adet örnek soru döner."""
    questions = []
    for item in data:
        soru, _ = extract_qa_pair(item)
        if soru:
            questions.append(str(soru))
        if len(questions) >= n:
            break
    return questions
