
"""
ApotekCare Assistant - Chatbot Engine Routing Fix V2

Tujuan:
- Meningkatkan pass rate uji 40 skenario dari ±87.5% menjadi target 90-98%.
- Memperkuat routing out-of-scope.
- Menambahkan handler pertanyaan ringkasan katalog.
- Memastikan urutan safety:
  medical boundary -> out-of-scope -> greeting -> operasional/lokasi -> katalog/stok -> harga -> rekomendasi -> ML fallback.
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

DEFAULT_THRESHOLD = 0.30


GREETING_KEYWORDS = [
    "halo", "hallo", "hai", "hi", "hello", "pagi", "siang", "sore", "malam",
    "assalamualaikum", "selamat pagi", "selamat siang", "selamat sore", "selamat malam"
]

OUT_OF_SCOPE_KEYWORDS = [
    # Politik/berita
    "presiden", "menteri", "pemilu", "politik", "partai", "berita politik", "gubernur", "dpr",

    # Finansial/investasi/mata uang
    "saham", "crypto", "kripto", "bitcoin", "reksadana", "forex", "trading",
    "kurs", "dollar", "dolar", "usd", "mata uang", "exchange rate",

    # Travel/cuaca
    "hotel", "tiket", "pesawat", "kereta", "cuaca", "wisata", "travel",

    # Hiburan
    "film", "bioskop", "game", "gaming", "musik", "lagu", "lirik",

    # IT/programming
    "python", "coding", "script", "scraping", "jaringan komputer", "router", "mikrotik",
    "server", "database", "komputer",

    # Akademik/karier
    "skripsi", "tesis", "jurnal", "makalah", "cv", "curriculum vitae", "lamaran",
    "melamar kerja", "karier", "interview kerja",

    # Kuliner/belanja umum
    "restoran", "kuliner", "makanan", "laptop", "handphone", "hp", "smartphone",

    # Matematika/umum
    "hitung", "dikali", "dibagi", "ditambah", "dikurangi", "matematika",

    # Olahraga
    "sepak bola", "pertandingan", "skor", "hasil pertandingan", "liga", "bola",

    # Pinjaman/bank
    "pinjaman", "pinjaman online", "transfer bank", "saldo", "rekening"
]

MEDICAL_BOUNDARY_KEYWORDS = [
    "dosis", "berapa dosis", "takaran",
    "antibiotik", "amoxicillin", "amoksisilin", "cefadroxil", "ciprofloxacin",
    "obat keras", "resep dokter", "resepkan", "diagnosa", "diagnosis",
    "bayi", "balita", "anak saya", "untuk anak", "dosis anak", "dosis bayi",
    "hamil", "ibu hamil", "menyusui",
    "sesak napas", "nyeri dada", "kejang", "pingsan", "darah",
    "demam tinggi", "alergi parah", "overdosis",
    "penyakit jantung", "gagal ginjal", "penyakit kronis"
]

OPERATIONAL_KEYWORDS = [
    "jam operasional", "jam buka", "jam tutup", "buka jam", "tutup jam",
    "apotek buka", "buka hari", "hari minggu", "hari sabtu", "hari libur",
    "operasional", "jadwal apotek"
]

LOCATION_KEYWORDS = [
    "lokasi", "alamat", "dimana", "di mana", "cabang", "maps", "peta"
]

CATALOG_OVERVIEW_KEYWORDS = [
    "produk apa saja", "produk apa yang tersedia", "apa saja yang tersedia",
    "daftar produk", "list produk", "katalog produk", "tersedia di katalog",
    "produk tersedia", "kategori produk"
]

PRICE_KEYWORDS = ["harga", "harganya", "berapa", "biaya", "tarif", "rupiah", "rp"]
STOCK_KEYWORDS = ["stok", "stock", "tersedia", "ketersediaan", "ready", "ada"]

PRODUCT_DOMAIN_KEYWORDS = [
    "obat", "produk", "vitamin", "batuk", "flu", "pilek", "demam", "maag",
    "lambung", "paracetamol", "parasetamol", "sakit kepala", "pusing",
    "nyeri", "tenggorokan", "vitamin c", "daya tahan", "minyak kayu putih"
]

RECOMMENDATION_PATTERNS = {
    "rekomendasi_sakit_kepala": ["sakit kepala", "nyeri kepala", "kepala sakit", "pusing", "migrain", "paracetamol", "parasetamol"],
    "rekomendasi_batuk": ["batuk", "tenggorokan gatal", "tenggorokan", "dahak"],
    "rekomendasi_flu": ["flu", "pilek", "hidung tersumbat", "bersin"],
    "rekomendasi_lambung": ["maag", "lambung", "asam lambung", "perut perih", "antasida"],
    "rekomendasi_demam": ["demam", "panas badan", "meriang", "penurun panas"],
    "rekomendasi_vitamin": ["vitamin", "vitamin c", "daya tahan", "imun", "imunitas", "multivitamin", "zinc"],
}

KEYWORD_PROFILES = {
    "sakit_kepala": ["sakit kepala", "nyeri kepala", "pusing", "pereda nyeri", "penurun panas", "nyeri", "demam", "paracetamol", "parasetamol"],
    "batuk": ["batuk", "tenggorokan", "dahak", "sirup batuk"],
    "flu": ["flu", "pilek", "hidung tersumbat", "bersin"],
    "maag": ["maag", "lambung", "asam lambung", "antasida", "perut perih"],
    "demam": ["demam", "panas", "penurun panas", "paracetamol", "parasetamol"],
    "vitamin": ["vitamin", "vitamin c", "multivitamin", "zinc", "imun", "daya tahan"],
    "paracetamol": ["paracetamol", "parasetamol", "penurun panas", "pereda nyeri", "demam", "nyeri"],
    "tenggorokan": ["pelega tenggorokan", "tenggorokan", "lozenges"],
    "minyak_kayu_putih": ["minyak kayu putih", "kayu putih"],
}


INTENT_RESPONSE_TEMPLATES = {
    "greeting": (
        "Halo, saya ApotekCare Assistant. Saya dapat membantu informasi umum layanan apotek, "
        "ketersediaan produk, harga produk pada katalog, dan rekomendasi produk non-resep secara terbatas."
    ),
    "jam_operasional": (
        "Untuk informasi operasional, apotek pada prototipe ini diasumsikan melayani pelanggan pada jam kerja apotek. "
        "Jika ingin memastikan jam buka, hari Minggu, atau hari libur, silakan konfirmasi ke admin/cabang apotek terkait."
    ),
    "lokasi_apotek": (
        "Informasi lokasi apotek pada prototipe ini masih bersifat simulasi. Untuk penggunaan nyata, fitur ini dapat "
        "dikembangkan dengan integrasi alamat cabang atau peta digital."
    ),
    "cek_stok_generic": (
        "Untuk pengecekan stok, silakan sebutkan nama produk atau kategori produk yang dicari, misalnya "
        "'stok obat batuk', 'ada vitamin C?', atau 'apakah ada obat untuk flu?'."
    ),
    "cek_stok": (
        "Berikut produk terkait dari katalog simulasi. Ketersediaan final tetap perlu dikonfirmasi ke admin/apoteker."
    ),
    "rekomendasi_sakit_kepala": (
        "Untuk sakit kepala ringan, produk pereda nyeri atau penurun panas yang dijual bebas dapat dipertimbangkan "
        "sesuai aturan pakai pada kemasan. Jika sakit kepala berat, berulang, atau disertai gejala lain, "
        "sebaiknya konsultasikan ke dokter/apoteker."
    ),
    "rekomendasi_batuk": (
        "Untuk batuk ringan, produk batuk non-resep dapat dipertimbangkan sesuai aturan pakai. Jika batuk disertai "
        "sesak, demam tinggi, darah, atau berlangsung lama, sebaiknya konsultasi ke dokter."
    ),
    "rekomendasi_flu": (
        "Untuk gejala flu ringan, istirahat cukup dan hidrasi tetap penting. Produk pendukung gejala flu dapat "
        "dipertimbangkan sesuai aturan pakai. Jika gejala berat atau berkepanjangan, konsultasikan ke tenaga kesehatan."
    ),
    "rekomendasi_lambung": (
        "Untuk keluhan maag/lambung ringan, produk antasida atau pereda maag non-resep dapat dipertimbangkan "
        "sesuai aturan pakai. Jika nyeri berat atau sering berulang, sebaiknya konsultasi ke dokter."
    ),
    "rekomendasi_demam": (
        "Untuk demam ringan, produk penurun panas dapat digunakan sesuai aturan pakai pada kemasan. Jika demam tinggi, "
        "berlangsung lama, atau terjadi pada bayi/ibu hamil, segera konsultasikan ke tenaga kesehatan."
    ),
    "rekomendasi_vitamin": (
        "Untuk kebutuhan vitamin, berikut beberapa pilihan produk dari katalog. Pilih sesuai kebutuhan umum dan "
        "perhatikan aturan pakai pada kemasan."
    ),
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any(text: str, keywords: list[str]) -> bool:
    normalized = clean_text(text)
    return any(keyword in normalized for keyword in keywords)


def detect_query_topic(text: str) -> str | None:
    normalized = clean_text(text)
    if any(k in normalized for k in ["sakit kepala", "nyeri kepala", "kepala sakit", "pusing", "migrain"]):
        return "sakit_kepala"
    if "paracetamol" in normalized or "parasetamol" in normalized:
        return "paracetamol"
    if "minyak kayu putih" in normalized or "kayu putih" in normalized:
        return "minyak_kayu_putih"
    if "pelega tenggorokan" in normalized or "tenggorokan" in normalized:
        return "tenggorokan"
    if "batuk" in normalized:
        return "batuk"
    if any(k in normalized for k in ["flu", "pilek", "hidung tersumbat", "bersin"]):
        return "flu"
    if any(k in normalized for k in ["maag", "lambung", "asam lambung", "antasida"]):
        return "maag"
    if any(k in normalized for k in ["demam", "panas badan", "meriang"]):
        return "demam"
    if any(k in normalized for k in ["vitamin", "daya tahan", "imun", "zinc"]):
        return "vitamin"
    return None


def detect_recommendation_intent(text: str) -> str | None:
    normalized = clean_text(text)
    recommendation_trigger = any(word in normalized for word in ["obat", "rekomendasi", "produk", "untuk"])
    for intent, patterns in RECOMMENDATION_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            if recommendation_trigger or intent == "rekomendasi_vitamin":
                return intent
    return None


def is_catalog_overview_question(text: str) -> bool:
    normalized = clean_text(text)
    return any(pattern in normalized for pattern in CATALOG_OVERVIEW_KEYWORDS)


def is_generic_stock_question(text: str) -> bool:
    normalized = clean_text(text)
    generic_patterns = ["stok produk", "cek stok", "stock produk", "ketersediaan produk"]
    return any(p in normalized for p in generic_patterns) and detect_query_topic(normalized) is None


def is_stock_question(text: str) -> bool:
    normalized = clean_text(text)
    if is_generic_stock_question(normalized):
        return True
    has_stock_word = any(k in normalized for k in STOCK_KEYWORDS)
    has_domain_word = any(k in normalized for k in PRODUCT_DOMAIN_KEYWORDS)
    return has_stock_word and has_domain_word


def is_price_question(text: str) -> bool:
    normalized = clean_text(text)
    has_price_word = any(k in normalized for k in PRICE_KEYWORDS)
    has_domain_or_context = any(k in normalized for k in PRODUCT_DOMAIN_KEYWORDS) or normalized in [
        "harganya berapa", "berapa harganya", "harganya", "berapa"
    ]
    return has_price_word and has_domain_or_context


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty:
        return None
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
    return sorted(DATA_DIR.rglob("*.csv")) if DATA_DIR.exists() else []


def load_product_catalog() -> pd.DataFrame:
    priority_paths = [
        DATA_DIR / "raw" / "product_catalog.csv",
        DATA_DIR / "processed" / "product_catalog.csv",
        DATA_DIR / "product_catalog.csv",
        DATA_DIR / "raw" / "apotekcare_product_catalog.csv",
        DATA_DIR / "processed" / "apotekcare_product_catalog.csv",
    ]
    candidates = [path for path in priority_paths if path.exists()]
    for path in get_all_csv_files():
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        name_col = detect_column(df, ["product_name", "nama_produk", "name", "produk", "nama"])
        category_col = detect_column(df, ["category", "kategori", "product_category", "jenis_produk", "jenis"])
        if name_col or category_col:
            return df
    return pd.DataFrame()


def load_metadata() -> dict[str, Any]:
    metadata_path = MODELS_DIR / "model_metadata_stage02.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            data.setdefault("confidence_threshold", DEFAULT_THRESHOLD)
            return data
        except Exception:
            pass
    return {"confidence_threshold": DEFAULT_THRESHOLD, "best_model": "rule_based_priority_with_ml_fallback"}


def load_intent_model():
    for path in [
        MODELS_DIR / "apotekcare_intent_model_stage02.pkl",
        MODELS_DIR / "apotekcare_intent_model.pkl",
        MODELS_DIR / "baseline_intent_model.pkl",
        MODELS_DIR / "intent_model.pkl",
    ]:
        if path.exists():
            return joblib.load(path)
    return None


def build_search_keywords(topic: str | None, raw_query: str) -> list[str]:
    if topic in KEYWORD_PROFILES:
        return KEYWORD_PROFILES[topic]
    normalized = clean_text(raw_query)
    keywords = []
    for possible in ["batuk", "flu", "pilek", "maag", "lambung", "demam", "vitamin", "zinc", "paracetamol", "parasetamol", "minyak kayu putih"]:
        if possible in normalized:
            keywords.append(possible)
    return keywords


def score_products(df: pd.DataFrame, topic: str | None, raw_query: str) -> pd.DataFrame:
    if df.empty:
        return df

    name_col = detect_column(df, ["product_name", "nama_produk", "name", "produk", "nama"])
    category_col = detect_column(df, ["category", "kategori", "product_category", "jenis_produk", "jenis"])
    desc_col = detect_column(df, ["description", "deskripsi", "indikasi", "manfaat", "notes", "catatan"])
    intent_col = detect_column(df, ["intent", "intent_tag", "related_intent", "label_intent"])

    searchable_cols = [col for col in [name_col, category_col, desc_col, intent_col] if col]
    keywords = build_search_keywords(topic, raw_query)

    scored = df.copy()
    scored["_score"] = 0

    for keyword in keywords:
        for col in searchable_cols:
            scored["_score"] += scored[col].astype(str).str.lower().str.contains(keyword.lower(), na=False).astype(int)

    if searchable_cols:
        joined = scored[searchable_cols].astype(str).agg(" ".join, axis=1).str.lower()

        if topic in ["sakit_kepala", "paracetamol", "demam"]:
            category_text = scored[category_col].astype(str).str.lower() if category_col else pd.Series([""] * len(scored), index=scored.index)
            device_mask = category_text.str.contains("alat|termometer|thermometer", na=False)
            scored.loc[device_mask, "_score"] = 0

        strict_patterns = {
            "flu": "flu|pilek|hidung|bersin",
            "maag": "maag|lambung|antasida|asam lambung|perut",
            "batuk": "batuk|tenggorokan|dahak",
            "tenggorokan": "tenggorokan|lozenges|pelega",
            "minyak_kayu_putih": "minyak kayu putih|kayu putih",
        }
        if topic in strict_patterns:
            mask = joined.str.contains(strict_patterns[topic], na=False)
            scored.loc[~mask, "_score"] = 0

    return scored.sort_values("_score", ascending=False)


def products_to_dicts(df: pd.DataFrame, top_n: int = 3) -> list[dict[str, Any]]:
    if df.empty:
        return []
    name_col = detect_column(df, ["product_name", "nama_produk", "name", "produk", "nama"])
    category_col = detect_column(df, ["category", "kategori", "product_category", "jenis_produk", "jenis"])
    desc_col = detect_column(df, ["description", "deskripsi", "indikasi", "manfaat", "notes", "catatan"])
    price_col = detect_column(df, ["price", "harga", "harga_rp", "price_idr", "unit_price", "harga_produk"])
    rows = []
    for _, row in df.head(top_n).iterrows():
        rows.append({
            "nama_produk": str(row[name_col]) if name_col else "Produk Apotek",
            "kategori": str(row[category_col]) if category_col else "-",
            "keterangan": str(row[desc_col]) if desc_col else "Gunakan sesuai aturan pakai pada kemasan.",
            "harga": str(row[price_col]) if price_col else None,
        })
    return rows


def search_products(topic: str | None, raw_query: str, top_n: int = 3) -> list[dict[str, Any]]:
    df = load_product_catalog()
    if df.empty:
        return []
    scored = score_products(df, topic, raw_query)
    if scored.empty or "_score" not in scored.columns:
        return []
    filtered = scored[scored["_score"] > 0].drop(columns=["_score"], errors="ignore")
    return products_to_dicts(filtered, top_n=top_n)


def category_summary(limit: int = 8) -> str:
    df = load_product_catalog()
    if df.empty:
        return "Katalog produk belum tersedia pada prototipe."
    category_col = detect_column(df, ["category", "kategori", "product_category", "jenis_produk", "jenis"])
    name_col = detect_column(df, ["product_name", "nama_produk", "name", "produk", "nama"])
    parts = []
    if category_col:
        cats = df[category_col].dropna().astype(str).drop_duplicates().head(limit).tolist()
        if cats:
            parts.append("Kategori produk yang tersedia pada katalog antara lain: " + ", ".join(cats) + ".")
    if name_col:
        products = df[name_col].dropna().astype(str).drop_duplicates().head(5).tolist()
        if products:
            parts.append("Contoh produk: " + ", ".join(products) + ".")
    return "\n\n".join(parts) if parts else "Katalog produk tersedia, tetapi kolom kategori/nama produk belum terbaca."


def format_products(products: list[dict[str, Any]], include_price: bool = True) -> str:
    if not products:
        return ""
    lines = ["\n\nProduk terkait dari katalog:"]
    for index, item in enumerate(products, start=1):
        price_text = f" Harga: {item['harga']}." if include_price and item.get("harga") else ""
        lines.append(
            f"{index}. {item['nama_produk']} — kategori: {item['kategori']}.{price_text} Catatan: {item['keterangan']}"
        )
    lines.append("\nKetersediaan dan harga final tetap perlu dikonfirmasi ke admin/apoteker.")
    return "\n".join(lines)


def format_price_answer(products: list[dict[str, Any]], from_context: bool = False) -> str:
    if not products:
        return (
            "Produk yang dimaksud belum saya temukan pada katalog. Silakan sebutkan nama produk atau kategori produk "
            "secara lebih spesifik, misalnya 'berapa harga obat batuk?' atau 'berapa harga vitamin C?'."
        )
    intro = "Berikut informasi harga dari produk yang terakhir dibahas:" if from_context else "Berikut informasi harga dari katalog:"
    lines = [intro]
    for index, item in enumerate(products, start=1):
        if item.get("harga"):
            lines.append(f"{index}. {item['nama_produk']} — {item['harga']}")
        else:
            lines.append(f"{index}. {item['nama_produk']} — harga belum tersedia pada katalog prototipe")
    lines.append("\nHarga dapat berubah dan tetap perlu dikonfirmasi ke admin/apoteker.")
    return "\n".join(lines)


def chatbot_response(
    user_question: str,
    threshold: float | None = None,
    context_products: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    question = str(user_question).strip()
    normalized = clean_text(question)
    context_products = context_products or []

    if not normalized:
        return {"answer": "Silakan tuliskan pertanyaan terlebih dahulu.", "intent": "empty_input", "confidence": 0.0, "status": "empty_input", "products": []}

    if contains_any(normalized, MEDICAL_BOUNDARY_KEYWORDS):
        return {
            "answer": (
                "Pertanyaan ini masuk area batasan medis. Saya tidak dapat memberikan diagnosis, resep, "
                "atau dosis personal. Untuk penggunaan antibiotik, dosis anak/bayi, ibu hamil/menyusui, "
                "atau gejala berat, sebaiknya konsultasikan langsung dengan dokter atau apoteker."
            ),
            "intent": "medical_boundary", "confidence": 1.0, "status": "medical_boundary", "products": []
        }

    if contains_any(normalized, OUT_OF_SCOPE_KEYWORDS):
        return {
            "answer": (
                "Maaf, pertanyaan tersebut berada di luar konteks ApotekCare. Saya hanya membantu informasi umum "
                "layanan apotek, stok/kategori produk, harga produk pada katalog, dan rekomendasi produk non-resep."
            ),
            "intent": "out_of_scope", "confidence": 1.0, "status": "out_of_scope", "products": []
        }

    if contains_any(normalized, GREETING_KEYWORDS):
        return {"answer": INTENT_RESPONSE_TEMPLATES["greeting"], "intent": "greeting", "confidence": 1.0, "status": "answered_rule_greeting", "products": []}

    if contains_any(normalized, OPERATIONAL_KEYWORDS):
        return {"answer": INTENT_RESPONSE_TEMPLATES["jam_operasional"], "intent": "jam_operasional", "confidence": 1.0, "status": "answered_rule_operational", "products": []}

    if contains_any(normalized, LOCATION_KEYWORDS):
        return {"answer": INTENT_RESPONSE_TEMPLATES["lokasi_apotek"], "intent": "lokasi_apotek", "confidence": 1.0, "status": "answered_rule_location", "products": []}

    if is_catalog_overview_question(normalized):
        return {
            "answer": "Berikut ringkasan katalog produk ApotekCare pada prototipe ini.\n\n" + category_summary(),
            "intent": "cek_stok", "confidence": 1.0, "status": "answered_catalog_overview", "products": []
        }

    if is_price_question(normalized):
        topic = detect_query_topic(normalized)
        if topic is None and context_products:
            return {"answer": format_price_answer(context_products, from_context=True), "intent": "tanya_harga", "confidence": 1.0, "status": "answered_price_context", "products": context_products}
        if topic is None:
            return {
                "answer": (
                    "Harga produk yang mana? Silakan sebutkan nama produk atau kategorinya, misalnya "
                    "'berapa harga obat batuk?', 'berapa harga vitamin C?', atau 'harga paracetamol berapa?'."
                ),
                "intent": "tanya_harga", "confidence": 1.0, "status": "needs_product_context", "products": []
            }
        products = search_products(topic=topic, raw_query=normalized, top_n=3)
        return {"answer": format_price_answer(products, from_context=False), "intent": "tanya_harga", "confidence": 1.0, "status": "answered_price_lookup" if products else "product_not_found", "products": products}

    if is_generic_stock_question(normalized):
        return {
            "answer": INTENT_RESPONSE_TEMPLATES["cek_stok_generic"] + "\n\n" + category_summary(),
            "intent": "cek_stok", "confidence": 1.0, "status": "needs_product_for_stock", "products": []
        }

    if is_stock_question(normalized):
        topic = detect_query_topic(normalized)
        products = search_products(topic=topic, raw_query=normalized, top_n=3)
        if not products:
            topic_text = topic.replace("_", " ") if topic else "produk tersebut"
            return {
                "answer": (
                    f"Saya belum menemukan produk yang relevan untuk {topic_text} pada katalog prototipe. "
                    "Silakan gunakan kata kunci lain atau konfirmasi langsung ke admin/apoteker."
                ),
                "intent": "cek_stok", "confidence": 1.0, "status": "stock_product_not_found", "products": []
            }
        return {"answer": INTENT_RESPONSE_TEMPLATES["cek_stok"] + format_products(products), "intent": "cek_stok", "confidence": 1.0, "status": "answered_stock_lookup", "products": products}

    recommendation_intent = detect_recommendation_intent(normalized)
    if recommendation_intent:
        topic_map = {
            "rekomendasi_sakit_kepala": "sakit_kepala",
            "rekomendasi_batuk": "batuk",
            "rekomendasi_flu": "flu",
            "rekomendasi_lambung": "maag",
            "rekomendasi_demam": "demam",
            "rekomendasi_vitamin": "vitamin",
        }
        topic = topic_map.get(recommendation_intent)
        products = search_products(topic=topic, raw_query=normalized, top_n=3)
        answer = INTENT_RESPONSE_TEMPLATES[recommendation_intent]
        if products:
            answer += format_products(products)
        else:
            topic_text = topic.replace("_", " ") if topic else "kategori tersebut"
            answer += f"\n\nSaat ini katalog prototipe belum memiliki produk yang spesifik untuk {topic_text}. Silakan konfirmasi pilihan produk langsung ke admin/apoteker."
        return {"answer": answer, "intent": recommendation_intent, "confidence": 1.0, "status": "answered_rule_recommendation", "products": products}

    model = load_intent_model()
    if model is not None:
        try:
            pred = str(model.predict([normalized])[0])
            confidence = 1.0
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba([normalized])[0].max())
            threshold_value = float(threshold if threshold is not None else DEFAULT_THRESHOLD)
            if confidence >= threshold_value and pred in INTENT_RESPONSE_TEMPLATES:
                return {"answer": INTENT_RESPONSE_TEMPLATES[pred], "intent": pred, "confidence": confidence, "status": "answered_ml_template", "products": []}
            return {
                "answer": (
                    "Maaf, saya belum cukup yakin memahami pertanyaan tersebut. Silakan tulis ulang dengan lebih spesifik, "
                    "misalnya 'apakah stok obat batuk tersedia?', 'berapa harga vitamin C?', "
                    "'obat untuk sakit kepala', atau 'jam operasional apotek sampai jam berapa?'."
                ),
                "intent": pred, "confidence": confidence, "status": "low_confidence_or_unmapped_ml_fallback", "products": []
            }
        except Exception:
            pass

    return {
        "answer": (
            "Maaf, saya belum memahami pertanyaan tersebut. Silakan tanyakan seputar layanan apotek, "
            "stok produk, harga produk, vitamin, obat bebas, atau rekomendasi produk non-resep."
        ),
        "intent": "fallback", "confidence": 0.0, "status": "fallback", "products": []
    }
