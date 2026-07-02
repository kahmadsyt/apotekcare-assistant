"""Pengujian chatbot melalui terminal."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.chatbot_engine import chatbot_response


questions = [
    "Apotek buka jam berapa?",
    "Saya batuk pilek, produk apa yang tersedia?",
    "Ada vitamin untuk daya tahan tubuh?",
    "Saya ingin antibiotik tanpa resep",
    "Tolong buatkan puisi cinta",
]

for q in questions:
    result = chatbot_response(q)
    print("=" * 80)
    print("Pertanyaan:", q)
    print("Intent:", result["intent"])
    print("Confidence:", result["confidence"])
    print("Jawaban:", result["response"])
