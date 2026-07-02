"""
Streamlit App TAHAP 02 - ApotekCare Assistant

Jalankan:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Agar import tetap aman saat streamlit dijalankan dari root project
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.chatbot_engine import chatbot_response, load_metadata  # noqa: E402

st.set_page_config(
    page_title="ApotekCare Assistant",
    page_icon="💊",
    layout="centered",
)

st.title("💊 ApotekCare Assistant")
st.caption("Chatbot layanan apotek berbasis NLP dan Text Mining untuk UAS Data Mining.")

with st.expander("Panduan singkat penggunaan", expanded=False):
    st.write(
        """
        Contoh pertanyaan:
        - Apakah ada rekomendasi vitamin untuk daya tahan tubuh?
        - Apakah stok obat batuk tersedia?
        - Jam operasional apotek sampai jam berapa?
        - Saya flu ringan, produk apa yang bisa dibeli bebas?

        Batasan:
        - Chatbot tidak memberikan diagnosis, resep dokter, atau dosis personal.
        - Untuk kondisi berat, bayi, ibu hamil/menyusui, antibiotik, atau penyakit kronis,
          konsultasikan langsung dengan dokter atau apoteker.
        """
    )

metadata = load_metadata()
default_threshold = float(metadata.get("confidence_threshold", 0.45))

with st.sidebar:
    st.header("Pengaturan")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=default_threshold,
        step=0.05,
        help="Semakin tinggi threshold, chatbot semakin berhati-hati dan lebih sering fallback.",
    )
    st.caption(f"Model aktif: {metadata.get('best_model', 'baseline/stage02')}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo, saya ApotekCare Assistant. Ada yang bisa saya bantu seputar layanan apotek atau rekomendasi produk non-resep?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Tulis pertanyaan Anda di sini...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    result = chatbot_response(question, threshold=threshold)
    answer = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(
            f"Intent: `{result['intent']}` | Confidence: `{result['confidence']:.3f}` | Status: `{result['status']}`"
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("Reset Percakapan"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Percakapan telah direset. Silakan ajukan pertanyaan baru seputar ApotekCare.",
        }
    ]
    st.rerun()
