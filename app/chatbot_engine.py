"""Engine chatbot ApotekCare Assistant."""

from pathlib import Path
import sys
import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.preprocessing import preprocess_text
from app.recommender import recommend_products


MODEL_PATH = ROOT_DIR / "models" / "intent_model.pkl"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ROOT_DIR / "models" / "label_encoder.pkl"
FAQ_PATH = ROOT_DIR / "data" / "raw" / "faq_apotekcare.csv"


RESPONSE_MAP = {
    "greeting": "Halo, saya ApotekCare Assistant. Saya dapat membantu informasi layanan apotek, cek stok, dan rekomendasi kategori produk kesehatan umum.",
    "info_jam_operasional": "ApotekCare buka setiap hari pukul 08.00 sampai 21.00. Untuk hari libur nasional, jam operasional dapat menyesuaikan.",
    "info_pemesanan": "Pelanggan dapat memesan produk melalui chat dengan menyebutkan nama produk atau kategori kebutuhan. Staf apotek akan membantu konfirmasi ketersediaan dan proses pemesanan.",
    "info_pengiriman": "ApotekCare menyediakan layanan antar untuk area tertentu. Biaya dan estimasi pengiriman akan dikonfirmasi oleh staf apotek.",
    "info_pembayaran": "Metode pembayaran yang tersedia meliputi tunai, transfer bank, QRIS, dan metode pembayaran digital yang tersedia di apotek.",
    "info_resep": "Untuk obat resep, pelanggan perlu menunjukkan atau mengirimkan resep dokter. Resep akan diverifikasi oleh apoteker atau staf apotek sesuai ketentuan.",
    "konsultasi_apoteker": "Untuk pertanyaan penggunaan obat, efek samping, interaksi obat, ibu hamil, anak-anak, atau pasien dengan kondisi khusus, sebaiknya berkonsultasi langsung dengan apoteker atau dokter.",
    "cek_stok_produk": "Bisa. Silakan tuliskan nama produk yang ingin dicek, misalnya vitamin C, masker medis, termometer digital, atau obat batuk bebas.",
    "info_promo": "Informasi promo dapat berubah sesuai periode. Silakan tanyakan produk yang diminati agar staf apotek dapat membantu mengecek promo yang tersedia.",
    "batasan_medis": "Maaf, ApotekCare Assistant tidak melakukan diagnosis medis dan tidak memberikan rekomendasi obat keras secara otomatis. Untuk kondisi khusus, obat resep, antibiotik, ibu hamil, bayi, atau gejala berat, silakan berkonsultasi dengan dokter atau apoteker.",
    "out_of_scope": "Maaf, pertanyaan tersebut di luar layanan ApotekCare. Saya hanya membantu informasi layanan apotek, cek stok, dan rekomendasi kategori produk kesehatan umum.",
}

RECOMMENDATION_INTENTS = {
    "rekomendasi_batuk_flu",
    "rekomendasi_demam_nyeri",
    "rekomendasi_vitamin",
    "rekomendasi_pencernaan",
    "rekomendasi_alat_kesehatan",
    "rekomendasi_produk_bayi",
    "rekomendasi_perawatan_luka",
}


def load_artifacts():
    """Load model, vectorizer, dan label encoder hasil training."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Model belum ditemukan. Jalankan: python scripts/train_model.py dari root project."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, vectorizer, label_encoder


def predict_intent(user_text: str):
    """Prediksi intent dari input user."""
    model, vectorizer, label_encoder = load_artifacts()
    clean_text = preprocess_text(user_text)
    X = vectorizer.transform([clean_text])
    pred_encoded = model.predict(X)[0]
    intent = label_encoder.inverse_transform([pred_encoded])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X).max())

    return intent, confidence, clean_text


def build_recommendation_response(intent: str) -> str:
    """Membangun jawaban rekomendasi produk berdasarkan intent."""
    product_df = recommend_products(intent, top_n=3)
    if product_df.empty:
        return RESPONSE_MAP.get(intent, "Maaf, data rekomendasi produk belum tersedia.")

    lines = [
        "Berikut kategori produk umum yang dapat dipertimbangkan:",
    ]

    for _, row in product_df.iterrows():
        lines.append(
            f"- {row['product_name']} ({row['category']}): {row['description']} Catatan: {row['safety_note']}"
        )

    lines.append(
        "\nCatatan: rekomendasi ini bersifat informasi umum, bukan diagnosis medis. Bacalah aturan pakai dan konsultasikan dengan apoteker bila memiliki kondisi khusus."
    )
    return "\n".join(lines)


def chatbot_response(user_text: str, confidence_threshold: float = 0.10) -> dict:
    """Menghasilkan respons chatbot lengkap dengan intent dan confidence."""
    intent, confidence, clean_text = predict_intent(user_text)

    if confidence is not None and confidence < confidence_threshold:
        intent = "out_of_scope"
        response = RESPONSE_MAP[intent]
    elif intent in RECOMMENDATION_INTENTS:
        response = build_recommendation_response(intent)
    else:
        response = RESPONSE_MAP.get(
            intent,
            "Maaf, saya belum memiliki jawaban untuk pertanyaan tersebut. Silakan konsultasikan dengan staf apotek."
        )

    return {
        "input_text": user_text,
        "clean_text": clean_text,
        "intent": intent,
        "confidence": confidence,
        "response": response,
    }
