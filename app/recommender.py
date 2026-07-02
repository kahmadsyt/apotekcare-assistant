"""Rekomendasi produk sederhana berbasis intent."""

from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PRODUCT_PATH = ROOT_DIR / "data" / "raw" / "product_catalog.csv"


def load_product_catalog() -> pd.DataFrame:
    """Load katalog produk apotek."""
    return pd.read_csv(PRODUCT_PATH)


def recommend_products(intent: str, top_n: int = 3) -> pd.DataFrame:
    """
    Mengambil rekomendasi produk berdasarkan intent hasil klasifikasi.
    Rekomendasi ini hanya bersifat kategori umum, bukan diagnosis medis.
    """
    product_df = load_product_catalog()
    result = product_df[product_df["related_intent"] == intent].head(top_n).copy()
    return result
