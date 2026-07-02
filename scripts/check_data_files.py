"""
Diagnostic script untuk mengecek file dataset ApotekCare secara recursive.

Jalankan dari root project:
    python scripts/check_data_files.py
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

print("=" * 80)
print("CEK DATASET APOTEKCARE - RECURSIVE")
print("=" * 80)
print("ROOT_DIR:", ROOT_DIR)
print("DATA_DIR :", DATA_DIR)

if not DATA_DIR.exists():
    print("[ERROR] Folder data/ tidak ditemukan.")
    raise SystemExit(1)

csv_files = sorted(DATA_DIR.rglob("*.csv"))

if not csv_files:
    print("[ERROR] Tidak ada file CSV di folder data/ maupun subfoldernya.")
    raise SystemExit(1)

for path in csv_files:
    print("-" * 80)
    print("File:", path.relative_to(ROOT_DIR))
    try:
        df = pd.read_csv(path)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        print("Preview:")
        print(df.head(3).to_string(index=False))
    except Exception as exc:
        print("[ERROR] Gagal membaca file:", exc)
