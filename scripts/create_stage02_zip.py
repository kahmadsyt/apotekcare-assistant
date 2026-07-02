"""
Membuat ZIP final TAHAP 02 ApotekCare Assistant.

Jalankan dari root project:
    python scripts/create_stage02_zip.py

Output:
    artifacts/TAHAP_02_APOTEKCARE_ASSISTANT_FINAL.zip
"""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "TAHAP_02_APOTEKCARE_ASSISTANT_FINAL.zip"
ZIP_PATH = ARTIFACTS_DIR / ZIP_NAME

INCLUDE_DIRS = [
    "app",
    "data",
    "models",
    "notebooks",
    "reports",
    "scripts",
]

INCLUDE_FILES = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "requirements.txt",
    "run_app.bat",
    "run_train.bat",
]

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "artifacts",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}


def should_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_PARTS:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def add_path_to_zip(zipf: zipfile.ZipFile, path: Path):
    if not path.exists():
        print(f"[SKIP] Tidak ditemukan: {path.relative_to(ROOT_DIR)}")
        return

    if path.is_file():
        if not should_exclude(path.relative_to(ROOT_DIR)):
            arcname = path.relative_to(ROOT_DIR)
            zipf.write(path, arcname)
            print(f"[ADD]  {arcname}")
        return

    for file_path in path.rglob("*"):
        if file_path.is_file() and not should_exclude(file_path.relative_to(ROOT_DIR)):
            arcname = file_path.relative_to(ROOT_DIR)
            zipf.write(file_path, arcname)
            print(f"[ADD]  {arcname}")


def write_manifest(zipf: zipfile.ZipFile):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest = f"""# Manifest ZIP TAHAP 02 - ApotekCare Assistant

Dibuat pada: {now}

Isi utama ZIP:
- Source code aplikasi Streamlit pada folder app/
- Dataset pada folder data/
- Model baseline/stage02 pada folder models/
- Notebook eksplorasi/evaluasi pada folder notebooks/
- Report evaluasi dan visualisasi pada folder reports/
- Script training, evaluasi, dan pembuatan ZIP pada folder scripts/
- README, requirements, dan file pendukung root project

Catatan:
ZIP ini dibuat untuk kebutuhan lampiran UAS Data Mining.
"""
    zipf.writestr("MANIFEST_TAHAP_02.md", manifest)


def main():
    print("=" * 80)
    print("MEMBUAT ZIP FINAL TAHAP 02 APOTEKCARE ASSISTANT")
    print("=" * 80)
    print(f"Project root : {ROOT_DIR}")
    print(f"Output ZIP   : {ZIP_PATH}")
    print("-" * 80)

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for directory in INCLUDE_DIRS:
            add_path_to_zip(zipf, ROOT_DIR / directory)

        for file_name in INCLUDE_FILES:
            add_path_to_zip(zipf, ROOT_DIR / file_name)

        write_manifest(zipf)

    print("-" * 80)
    print(f"[SUCCESS] ZIP final berhasil dibuat: {ZIP_PATH}")
    print(f"[INFO] Ukuran file: {ZIP_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
