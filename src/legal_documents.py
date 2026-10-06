"""
Kanun ve mevzuat belgelerinin yönetimi.
data/mevzuat/ klasöründeki .txt ve .json dosyalarını okur.
"""

import os
import json
import glob
from typing import List, Dict

import config


def list_documents() -> List[str]:
    """Mevzuat dizinindeki belge dosyalarını listeler."""
    patterns = ["*.txt", "*.json", "*.md"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(config.MEVZUAT_DIR, pattern)))
    return sorted(files)


def read_document(file_path: str) -> str:
    """Bir belge dosyasının içeriğini okur."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Belge bulunamadı: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_document(filename: str, content: str) -> str:
    """Yeni bir mevzuat belgesini kaydeder."""
    os.makedirs(config.MEVZUAT_DIR, exist_ok=True)
    file_path = os.path.join(config.MEVZUAT_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path


def search_in_documents(query: str) -> List[Dict[str, str]]:
    """
    Mevzuat belgelerinde basit metin araması yapar.
    Eşleşen satırları dosya adıyla birlikte döner.
    """
    results = []
    query_lower = query.lower()

    for file_path in list_documents():
        try:
            content = read_document(file_path)
            filename = os.path.basename(file_path)

            for line_no, line in enumerate(content.splitlines(), 1):
                if query_lower in line.lower():
                    results.append({
                        "dosya": filename,
                        "satir": line_no,
                        "icerik": line.strip(),
                    })
        except Exception:
            continue

    return results
