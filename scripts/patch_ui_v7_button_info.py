from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_FILE = ROOT_DIR / "app" / "app.py"
BACKUP_FILE = ROOT_DIR / "app" / "app_backup_before_ui_v7_patch.py"

if not APP_FILE.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {APP_FILE}")

text = APP_FILE.read_text(encoding="utf-8")

if not BACKUP_FILE.exists():
    BACKUP_FILE.write_text(text, encoding="utf-8")
    print(f"[BACKUP] Backup dibuat: {BACKUP_FILE}")
else:
    print(f"[INFO] Backup sudah tersedia: {BACKUP_FILE}")

css_patch = r"""
/* =========================================================
   PATCH V7 - Button text and info/error-analysis readability
   ========================================================= */

div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button:focus,
div[data-testid="stFormSubmitButton"] button:active,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #0f766e !important;
    background-color: #0f766e !important;
    border: 1px solid #0f766e !important;
    color: #ffffff !important;
    opacity: 1 !important;
    filter: none !important;
    box-shadow: 0 6px 14px rgba(15,118,110,.22) !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #0d5f59 !important;
    background-color: #0d5f59 !important;
    border-color: #0d5f59 !important;
}

div[data-testid="stFormSubmitButton"] button *,
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span,
div[data-testid="stFormSubmitButton"] button div {
    color: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.stButton button,
.stButton button:focus,
.stButton button:active,
.stButton button:hover {
    color: #ffffff !important;
    opacity: 1 !important;
}

.stButton button *,
.stButton button p,
.stButton button span,
.stButton button div {
    color: #ffffff !important;
    opacity: 1 !important;
    visibility: visible !important;
}

.info-box {
    background: #eaf4ff !important;
    background-color: #eaf4ff !important;
    border: 1px solid #93c5fd !important;
    border-left: 6px solid #0284c7 !important;
    border-radius: 18px !important;
    padding: .95rem 1rem !important;
    margin: .9rem 0 1rem 0 !important;
    color: #0f172a !important;
    line-height: 1.55 !important;
    opacity: 1 !important;
}

.info-box,
.info-box *,
.info-box p,
.info-box span,
.info-box div,
.info-box li,
.info-box b,
.info-box strong {
    color: #0f172a !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stMarkdownContainer"] .info-box,
div[data-testid="stMarkdownContainer"] .info-box * {
    color: #0f172a !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stExpander"],
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background: rgba(255,255,255,.98) !important;
    color: #0f172a !important;
}

div[data-testid="stExpander"] *,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] div {
    color: #0f172a !important;
    opacity: 1 !important;
    visibility: visible !important;
}
"""

if "PATCH V7 - Button text and info/error-analysis readability" not in text:
    if "</style>" not in text:
        raise RuntimeError("Tag </style> tidak ditemukan pada app/app.py. Patch CSS tidak dapat diterapkan otomatis.")
    text = text.replace("</style>", css_patch + "\n</style>", 1)
    print("[INFO] CSS Patch V7 ditambahkan.")
else:
    print("[INFO] CSS Patch V7 sudah ada. Tidak menambahkan ulang CSS.")

APP_FILE.write_text(text, encoding="utf-8")

print("[SUCCESS] Patch UI V7 berhasil diterapkan.")
print("[NEXT] Jalankan: streamlit run app/app.py")