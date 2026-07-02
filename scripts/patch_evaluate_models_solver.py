"""
Patch Logistic Regression solver untuk multiclass classification.

Masalah:
- solver='liblinear' tidak mendukung multiclass pada environment scikit-learn Anda.
- Dataset ApotekCare memiliki intent lebih dari 2 kelas.
- Solusi: ganti solver Logistic Regression menjadi 'lbfgs'.

Cara menjalankan dari root project:
    python scripts/patch_evaluate_models_solver.py

Setelah patch:
    python scripts/evaluate_models.py
"""

from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_FILE = ROOT_DIR / "scripts" / "evaluate_models.py"
BACKUP_FILE = ROOT_DIR / "scripts" / "evaluate_models_before_multiclass_solver_fix.py"

if not TARGET_FILE.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {TARGET_FILE}")

text = TARGET_FILE.read_text(encoding="utf-8")

# Backup file lama
if not BACKUP_FILE.exists():
    BACKUP_FILE.write_text(text, encoding="utf-8")
    print(f"[BACKUP] File backup dibuat: {BACKUP_FILE}")
else:
    print(f"[INFO] Backup sudah ada: {BACKUP_FILE}")

original_text = text

# Pola 1: solver="liblinear"
text = text.replace('solver="liblinear"', 'solver="lbfgs"')

# Pola 2: solver='liblinear'
text = text.replace("solver='liblinear'", "solver='lbfgs'")

# Tambahan: jika max_iter masih kecil, naikkan agar lebih aman untuk TF-IDF multiclass.
text = re.sub(r"max_iter\s*=\s*1000", "max_iter=2000", text)

if text == original_text:
    print("[WARN] Tidak ada teks solver='liblinear' atau solver=\"liblinear\" yang ditemukan.")
    print("[INFO] Silakan cek manual fungsi build_models() pada scripts/evaluate_models.py.")
else:
    TARGET_FILE.write_text(text, encoding="utf-8")
    print("[SUCCESS] Patch berhasil diterapkan.")
    print("[INFO] LogisticRegression sekarang menggunakan solver='lbfgs'.")
    print("[INFO] max_iter diset menjadi 2000 jika sebelumnya 1000.")

print("\nLangkah berikutnya:")
print("    python scripts/evaluate_models.py")
