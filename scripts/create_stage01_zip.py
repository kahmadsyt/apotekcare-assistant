"""Membuat ZIP final TAHAP 01 ApotekCare Assistant."""

from pathlib import Path
import shutil

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_base = OUTPUT_DIR / "TAHAP_01_APOTEKCARE_DATASET_BASELINE"
zip_path = shutil.make_archive(str(output_base), "zip", root_dir=ROOT_DIR)

print(f"ZIP berhasil dibuat: {zip_path}")
