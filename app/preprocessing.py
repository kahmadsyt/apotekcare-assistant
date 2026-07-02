"""Utility preprocessing teks untuk ApotekCare Assistant."""

import re


def preprocess_text(text: str) -> str:
    """
    Membersihkan teks input user.

    Tahapan:
    1. Case folding
    2. Menghapus tanda baca dan karakter non huruf
    3. Normalisasi spasi
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
