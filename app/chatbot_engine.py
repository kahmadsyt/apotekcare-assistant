"""
Chatbot Engine TAHAP 02 - ApotekCare Assistant
Versi FIX RECURSIVE DATA PATH.

Perbaikan:
- Product catalog dapat dibaca dari:
  - data/raw/product_catalog.csv
  - data/processed/product_catalog.csv
  - data/product_catalog.csv
  - atau CSV lain pada data/** yang memiliki kolom kategori produk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_THRESHOLD = 0.45

OUT_OF_SCOPE_KEYWORDS = [
    "transfer bank",
    "pinjaman",
    "politik",
    "pemilu",
    "saham",
    "crypto",
    "game",
    "film",
    "cuaca",
    "hotel",
    "tiket pesawat",
    "coding",
    "python error",
    "jaringan komputer",
    "skripsi",
    "tesis",
]

MEDICAL_BOUNDARY_KEYWORDS = [
    "diagnosa",
    "diagnosis",
    "vonis",
    "resep dokter",
    "antibiotik",
    "obat keras",
    "dosis anak",
    "dosis bayi",
    "ibu hamil",
    "hamil",
    "menyusui",
    "bayi",
    "balita",
    "overdosis",
    "pingsan",
    "sesak napas",
    "nyeri dada",
    "kejang",
    "alergi parah",
    "demam tinggi",
    "darah",
    "infeksi berat",
    "penyakit kronis",
    "gagal ginjal",
    "penyakit jantung",
]

INTENT_RESPONSE_TEMPLATES = {
    "greeting": "Halo, saya ApotekCare Assistant. Saya dapat membantu informasi umum layanan apotek, ketersediaan produk, dan rekomendasi produk non-resep secara terbatas.",
    "jam_operasional": "Jam operasional apotek dapat disesuaikan dengan cabang. Secara umum, layanan tersedia pada jam kerja apotek. Untuk kepastian, silakan cek cabang terdekat atau hubungi admin apotek.",
    "cek_stok": "Untuk pengecekan stok, sebutkan nama produk atau kategori produk yang Anda cari. Saya akan membantu mencarikan rekomendasi berdasarkan katalog yang tersedia.",
    "lokasi_apotek": "Informasi lokasi apotek dapat dicek melalui cabang terdekat. Pada prototipe ini, fitur lokasi masih bersifat simulasi dan belum terhubung ke peta real-time.",
    "rekomendasi_vitamin": "Untuk kebutuhan vitamin, saya dapat menampilkan beberapa pilihan produk dari katalog. Pilih sesuai kebutuhan umum dan perhatikan aturan pakai pada kemasan.",
    "rekomendasi_batuk": "Untuk batuk ringan, saya dapat memberikan rekomendasi produk non-resep dari katalog. Bila batuk disertai sesak, demam tinggi, darah, atau berlangsung lama, sebaiknya konsultasi ke dokter.",
    "rekomendasi_flu": "Untuk gejala flu ringan, istirahat cukup dan hidrasi tetap penting. Saya dapat menampilkan produk pendukung dari katalog, tetapi hindari penggunaan obat bila memiliki kondisi khusus tanpa konsultasi tenaga kesehatan.",
    "rekomendasi_lambung": "Untuk keluhan lambung ringan, produk antasida atau pereda maag non-resep dapat dipertimbangkan sesuai aturan pakai. Bila nyeri berat atau berulang, sebaiknya konsultasi dengan dokter.",
    "rekomendasi_demam": "Untuk demam ringan, obat penurun panas umum dapat digunakan sesuai aturan pakai. Jika demam tinggi, berlangsung lama, atau terjadi pada bayi/ibu hamil, segera konsultasi dengan tenaga kesehatan.",
    "fallback": "Maaf, saya belum cukup yakin memahami pertanyaan tersebut. Silakan tanyakan seputar layanan apotek, stok produk, kategori obat bebas, vitamin, atau rekomendasi produk non-resep.",
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_metadata() -> dict[str, Any]:
    metadata_path = MODELS_DIR / "model_metadata_stage02.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"confidence_threshold": DEFAULT_THRESHOLD}


def load_intent_model():
    candidates = [
        MODELS_DIR / "apotekcare_intent_model_stage02.pkl",
        MODELS_DIR / "apotekcare_intent_model.pkl",
        MODELS_DIR / "baseline_intent_model.pkl",
        MODELS_DIR / "intent_model.pkl",
    ]
    for path in candidates:
        if path.exists():
            return joblib.load(path), path
    raise FileNotFoundError(
        "Model intent tidak ditemukan. Jalankan terlebih dahulu: python scripts/evaluate_models.py"
    )


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).lower().strip(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in df.columns:
        col_lower = str(col).lower().strip()
        for candidate in candidates:
            if candidate.lower() in col_lower:
                return col

    return None


def get_all_csv_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.rglob("*.csv"))


def load_product_catalog() -> pd.DataFrame:
    priority_relative_paths = [
        "data/raw/product_catalog.csv",
        "data/processed/product_catalog.csv",
        "data/product_catalog.csv",
        "data/raw/apotekcare_product_catalog.csv",
        "data/processed/apotekcare_product_catalog.csv",
        "data/raw/products.csv",
        "data/processed/products.csv",
    ]

    candidates: list[Path] = []
    for rel_path in priority_relative_paths:
        path = ROOT_DIR / rel_path
        if path.exists():
            candidates.append(path)

    for path in get_all_csv_files():
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        category_col = detect_column(
            df,
            ["category", "kategori", "product_category", "jenis_produk", "jenis", "tipe_produk"],
        )
        name_col = detect_column(
            df,
            ["product_name", "nama_produk", "name", "produk", "nama"],
        )

        if category_col or name_col:
            df.columns = [str(c).strip() for c in df.columns]
            return df

    return pd.DataFrame()


def contains_any(text: str, keywords: list[str]) -> bool:
    normalized = clean_text(text)
    return any(keyword in normalized for keyword in keywords)


def detect_out_of_scope(user_question: str) -> bool:
    return contains_any(user_question, OUT_OF_SCOPE_KEYWORDS)


def detect_medical_boundary(user_question: str) -> bool:
    return contains_any(user_question, MEDICAL_BOUNDARY_KEYWORDS)


def predict_intent(user_question: str, model=None) -> dict[str, Any]:
    if model is None:
        model, model_path = load_intent_model()
    else:
        model_path = None

    cleaned = clean_text(user_question)
    predicted_intent = model.predict([cleaned])[0]

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(probabilities.max())

    return {
        "intent": str(predicted_intent),
        "confidence": confidence,
        "model_path": str(model_path) if model_path else None,
        "cleaned_text": cleaned,
    }


def recommend_products(intent: str, top_n: int = 3) -> list[dict[str, Any]]:
    df = load_product_catalog()
    if df.empty:
        return []

    name_col = detect_column(df, ["product_name", "nama_produk", "name", "produk", "nama"])
    category_col = detect_column(df, ["category", "kategori", "product_category", "jenis_produk", "jenis", "tipe_produk"])
    intent_col = detect_column(df, ["intent", "intent_tag", "related_intent", "label_intent"])
    desc_col = detect_column(df, ["description", "deskripsi", "indikasi", "manfaat", "notes", "catatan"])

    filtered = df.copy()

    if intent_col:
        filtered = filtered[filtered[intent_col].astype(str).str.lower().str.contains(intent.lower(), na=False)]

    if filtered.empty and category_col:
        keyword = intent.replace("rekomendasi_", "").replace("_", " ")
        filtered = df[df[category_col].astype(str).str.lower().str.contains(keyword.lower(), na=False)]

    if filtered.empty:
        filtered = df.head(top_n)

    results = []
    for _, row in filtered.head(top_n).iterrows():
        results.append(
            {
                "nama_produk": str(row[name_col]) if name_col else "Produk Apotek",
                "kategori": str(row[category_col]) if category_col else "-",
                "keterangan": str(row[desc_col]) if desc_col else "Gunakan sesuai aturan pakai pada kemasan.",
            }
        )
    return results


def format_product_recommendations(products: list[dict[str, Any]]) -> str:
    if not products:
        return ""

    lines = ["\n\nRekomendasi produk dari katalog:"]
    for idx, item in enumerate(products, start=1):
        lines.append(
            f"{idx}. {item['nama_produk']} — kategori: {item['kategori']}. "
            f"Catatan: {item['keterangan']}"
        )
    lines.append("\nPastikan membaca aturan pakai dan konsultasikan dengan apoteker bila memiliki kondisi khusus.")
    return "\n".join(lines)


def chatbot_response(user_question: str, threshold: float | None = None) -> dict[str, Any]:
    metadata = load_metadata()
    threshold = float(threshold if threshold is not None else metadata.get("confidence_threshold", DEFAULT_THRESHOLD))

    if not str(user_question).strip():
        return {
            "answer": "Silakan tuliskan pertanyaan terlebih dahulu.",
            "intent": "empty_input",
            "confidence": 0.0,
            "status": "empty_input",
            "products": [],
        }

    if detect_out_of_scope(user_question):
        return {
            "answer": (
                "Maaf, pertanyaan tersebut berada di luar konteks ApotekCare. "
                "Saya hanya membantu informasi umum layanan apotek, stok/kategori produk, "
                "dan rekomendasi produk non-resep secara terbatas."
            ),
            "intent": "out_of_scope",
            "confidence": 1.0,
            "status": "out_of_scope",
            "products": [],
        }

    if detect_medical_boundary(user_question):
        return {
            "answer": (
                "Pertanyaan ini masuk area batasan medis. Saya tidak dapat memberikan diagnosis, "
                "resep, atau dosis personal. Untuk kondisi khusus seperti bayi, ibu hamil/menyusui, "
                "antibiotik, gejala berat, atau penyakit kronis, sebaiknya konsultasikan langsung "
                "dengan dokter atau apoteker."
            ),
            "intent": "medical_boundary",
            "confidence": 1.0,
            "status": "medical_boundary",
            "products": [],
        }

    model, _ = load_intent_model()
    prediction = predict_intent(user_question, model=model)
    intent = prediction["intent"]
    confidence = float(prediction["confidence"])

    if confidence < threshold:
        return {
            "answer": (
                "Maaf, saya belum cukup yakin memahami pertanyaan tersebut. "
                "Silakan tulis ulang dengan lebih spesifik, misalnya: "
                "'apakah ada vitamin untuk daya tahan tubuh?', "
                "'apakah stok obat batuk tersedia?', atau "
                "'jam operasional apotek sampai jam berapa?'."
            ),
            "intent": intent,
            "confidence": confidence,
            "status": "low_confidence_fallback",
            "products": [],
        }

    base_answer = INTENT_RESPONSE_TEMPLATES.get(
        intent,
        "Saya memahami pertanyaan Anda terkait layanan ApotekCare. Berikut jawaban umum yang dapat saya bantu.",
    )

    products = []
    if "rekomendasi" in intent or any(
        keyword in clean_text(user_question)
        for keyword in ["obat", "vitamin", "produk", "batuk", "flu", "demam", "maag", "lambung"]
    ):
        products = recommend_products(intent=intent, top_n=3)

    answer = base_answer + format_product_recommendations(products)

    return {
        "answer": answer,
        "intent": intent,
        "confidence": confidence,
        "status": "answered",
        "products": products,
    }
