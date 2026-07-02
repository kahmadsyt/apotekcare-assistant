"""Dashboard Streamlit ApotekCare Assistant."""

from pathlib import Path
import sys
import subprocess

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.chatbot_engine import chatbot_response, MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH


st.set_page_config(
    page_title="ApotekCare Assistant",
    page_icon="💊",
    layout="wide"
)

st.title("💊 ApotekCare Assistant")
st.caption("Chatbot layanan apotek berbasis NLP, TF-IDF, dan supervised learning.")

with st.sidebar:
    st.header("Panduan Penggunaan")
    st.write(
        "Tanyakan layanan apotek seperti jam operasional, pemesanan, pengiriman, pembayaran, "
        "cek stok, resep dokter, atau rekomendasi kategori produk umum."
    )
    st.warning(
        "Chatbot tidak melakukan diagnosis medis dan tidak merekomendasikan obat keras/antibiotik secara otomatis."
    )

    st.subheader("Contoh pertanyaan")
    st.markdown("- Apotek buka jam berapa?")
    st.markdown("- Saya batuk pilek, produk apa yang tersedia?")
    st.markdown("- Ada vitamin untuk daya tahan tubuh?")
    st.markdown("- Apakah bisa tebus resep dokter?")

    if st.button("Latih ulang model"):
        result = subprocess.run(
            [sys.executable, str(ROOT_DIR / "scripts" / "train_model.py")],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            st.success("Model berhasil dilatih ulang. Silakan refresh halaman jika diperlukan.")
        else:
            st.error("Model gagal dilatih ulang.")
            st.code(result.stderr)

# Latih otomatis jika model belum ada
if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists() and LABEL_ENCODER_PATH.exists()):
    with st.spinner("Model belum ditemukan. Melatih model baseline terlebih dahulu..."):
        result = subprocess.run(
            [sys.executable, str(ROOT_DIR / "scripts" / "train_model.py")],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            st.error("Model gagal dibuat. Jalankan manual: python scripts/train_model.py")
            st.code(result.stderr)
            st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Area Chat Interaktif")
    user_question = st.text_input("Masukkan pertanyaan pelanggan:", placeholder="Contoh: Saya demam ringan, produk apa yang tersedia?")

    if st.button("Kirim Pertanyaan") and user_question.strip():
        result = chatbot_response(user_question)
        st.markdown("### Jawaban Chatbot")
        st.write(result["response"])

        st.markdown("### Informasi Prediksi")
        st.write(f"**Intent:** `{result['intent']}`")
        if result["confidence"] is not None:
            st.write(f"**Confidence:** `{result['confidence']:.4f}`")
        st.write(f"**Teks setelah preprocessing:** `{result['clean_text']}`")

with col2:
    st.subheader("Dataset Ringkas")
    data_dir = ROOT_DIR / "data" / "raw"
    intent_df = pd.read_csv(data_dir / "intent_dataset.csv")
    faq_df = pd.read_csv(data_dir / "faq_apotekcare.csv")
    product_df = pd.read_csv(data_dir / "product_catalog.csv")

    st.metric("FAQ", len(faq_df))
    st.metric("Data Intent", len(intent_df))
    st.metric("Jumlah Intent", intent_df["intent"].nunique())
    st.metric("Produk/Kategori", len(product_df))

    st.markdown("### Distribusi Intent")
    st.bar_chart(intent_df["intent"].value_counts())
