"""
Menambahkan kolom harga simulasi pada product_catalog.csv jika belum ada.

Jalankan dari root project:
    python scripts/add_sample_prices_to_product_catalog.py

Catatan:
Harga ini hanya simulasi untuk kebutuhan prototipe UAS Data Mining.
Silakan edit manual agar sesuai kebutuhan laporan.
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

candidate_paths = [
    ROOT_DIR / "data" / "raw" / "product_catalog.csv",
    ROOT_DIR / "data" / "processed" / "product_catalog.csv",
    ROOT_DIR / "data" / "product_catalog.csv",
]

product_path = None
for path in candidate_paths:
    if path.exists():
        product_path = path
        break

if product_path is None:
    raise FileNotFoundError(
        "product_catalog.csv tidak ditemukan pada data/raw, data/processed, atau data/."
    )

df = pd.read_csv(product_path)

lower_cols = {str(c).lower().strip(): c for c in df.columns}
if "harga" in lower_cols:
    print(f"[INFO] Kolom harga sudah ada pada {product_path}. Tidak ada perubahan.")
    raise SystemExit(0)


def detect_col(candidates):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


name_col = detect_col(["product_name", "nama_produk", "name", "produk", "nama"])
category_col = detect_col(["category", "kategori", "product_category", "jenis_produk", "jenis"])


def assign_price(row):
    text_parts = []
    if name_col:
        text_parts.append(str(row.get(name_col, "")))
    if category_col:
        text_parts.append(str(row.get(category_col, "")))

    text = " ".join(text_parts).lower()

    if "vitamin" in text:
        return "Rp25.000"
    if "batuk" in text:
        return "Rp18.000"
    if "flu" in text or "pilek" in text:
        return "Rp20.000"
    if "maag" in text or "lambung" in text or "antasida" in text:
        return "Rp12.000"
    if "demam" in text or "panas" in text or "nyeri" in text or "paracetamol" in text:
        return "Rp10.000"
    if "termometer" in text or "thermometer" in text or "alat" in text:
        return "Rp45.000"
    return "Rp15.000"


df["harga"] = df.apply(assign_price, axis=1)

backup_path = product_path.with_suffix(".backup_before_price.csv")
if not backup_path.exists():
    pd.read_csv(product_path).to_csv(backup_path, index=False)

df.to_csv(product_path, index=False)

print("[SUCCESS] Kolom harga simulasi berhasil ditambahkan.")
print(f"[INFO] File katalog : {product_path}")
print(f"[INFO] Backup       : {backup_path}")
print("[INFO] Silakan cek kembali product_catalog.csv dan edit harga jika diperlukan.")
