"""
TAHAP 02 - Evaluasi Model, Error Analysis, dan Visualisasi ApotekCare Assistant
Versi FIX RECURSIVE DATA PATH.

Masalah yang diperbaiki:
- Dataset tidak berada langsung di data/, tetapi di:
  - data/raw/
  - data/processed/
- Script sekarang mencari CSV secara recursive menggunakan data/**/*.csv.

Jalankan dari root project:
    python scripts/evaluate_models.py

Prioritas dataset intent:
1. data/processed/intent_dataset_clean.csv
2. data/raw/intent_dataset.csv
3. file CSV lain yang memiliki kolom teks pertanyaan dan label intent

Output utama:
- reports/model_comparison_metrics.csv
- reports/classification_report_logistic_regression.csv
- reports/classification_report_multinomial_nb.csv
- reports/error_analysis_logistic_regression.csv
- reports/error_analysis_multinomial_nb.csv
- reports/figures/intent_distribution.png
- reports/figures/confusion_matrix_logistic_regression.png
- reports/figures/confusion_matrix_multinomial_nb.png
- reports/figures/model_metrics_comparison.png
- reports/figures/product_category_distribution.png
- models/apotekcare_intent_model_stage02.pkl
- models/model_metadata_stage02.json
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN_CANDIDATES = [
    "question",
    "questions",
    "pertanyaan",
    "text",
    "teks",
    "utterance",
    "utterances",
    "input_text",
    "kalimat",
    "user_question",
    "customer_question",
    "query",
    "sentence",
]

LABEL_COLUMN_CANDIDATES = [
    "intent",
    "label",
    "labels",
    "category",
    "kategori",
    "kelas",
    "class",
    "tag",
    "intent_label",
    "intent_name",
]

PRODUCT_CATEGORY_COLUMN_CANDIDATES = [
    "category",
    "kategori",
    "product_category",
    "jenis_produk",
    "jenis",
    "tipe_produk",
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_column(
    df: pd.DataFrame,
    candidates: list[str],
    semantic_name: str,
    raise_error: bool = True,
) -> str | None:
    lower_map = {str(col).lower().strip(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in df.columns:
        col_lower = str(col).lower().strip()
        for candidate in candidates:
            if candidate.lower() in col_lower:
                return col

    if raise_error:
        raise ValueError(
            f"Kolom {semantic_name} tidak ditemukan. "
            f"Gunakan salah satu nama kolom berikut: {candidates}. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    return None


def get_all_csv_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.rglob("*.csv"))


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def print_data_folder_diagnostics():
    print("\n[DIAGNOSTIC] ROOT_DIR :", ROOT_DIR)
    print("[DIAGNOSTIC] DATA_DIR :", DATA_DIR)

    if not DATA_DIR.exists():
        print("[ERROR] Folder data/ tidak ditemukan.")
        return

    csv_files = get_all_csv_files()
    if not csv_files:
        print("[ERROR] Tidak ada file .csv pada folder data/ maupun subfoldernya.")
        return

    print("\n[DIAGNOSTIC] File CSV yang ditemukan secara recursive:")
    for idx, path in enumerate(csv_files, start=1):
        try:
            sample = pd.read_csv(path, nrows=3)
            print(f"  {idx}. {relative_to_root(path)} | columns={list(sample.columns)}")
        except Exception as exc:
            print(f"  {idx}. {relative_to_root(path)} | gagal dibaca: {exc}")


def find_intent_dataset_file() -> tuple[Path, str, str]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Folder data/ tidak ditemukan: {DATA_DIR}")

    priority_relative_paths = [
        "data/processed/intent_dataset_clean.csv",
        "data/processed/intent_dataset.csv",
        "data/processed/apotekcare_intent_dataset.csv",
        "data/raw/intent_dataset.csv",
        "data/raw/apotekcare_intent_dataset.csv",
        "data/raw/apotekcare_intents.csv",
        "data/raw/faq_intent_dataset.csv",
        "data/raw/intent_classification_dataset.csv",
        "data/intent_dataset.csv",
        "data/apotekcare_intent_dataset.csv",
        "data/dataset_intent.csv",
        "data/faq_intent_dataset.csv",
    ]

    candidate_files: list[Path] = []
    for rel_path in priority_relative_paths:
        path = ROOT_DIR / rel_path
        if path.exists():
            candidate_files.append(path)

    for path in get_all_csv_files():
        if path not in candidate_files:
            candidate_files.append(path)

    if not candidate_files:
        print_data_folder_diagnostics()
        raise FileNotFoundError(f"Tidak ada file CSV di dalam folder data/: {DATA_DIR}")

    checked_info = []

    for path in candidate_files:
        try:
            df_sample = pd.read_csv(path, nrows=30)
        except Exception as exc:
            checked_info.append(f"- {relative_to_root(path)}: gagal dibaca ({exc})")
            continue

        text_col = detect_column(df_sample, TEXT_COLUMN_CANDIDATES, "teks pertanyaan", raise_error=False)
        label_col = detect_column(df_sample, LABEL_COLUMN_CANDIDATES, "label intent", raise_error=False)

        checked_info.append(
            f"- {relative_to_root(path)}: columns={list(df_sample.columns)} | "
            f"text_col={text_col} | label_col={label_col}"
        )

        if text_col is not None and label_col is not None:
            return path, text_col, label_col

    print_data_folder_diagnostics()
    raise FileNotFoundError(
        "Tidak ditemukan CSV yang memiliki kolom teks pertanyaan dan label intent.\n\n"
        "Pastikan dataset intent memiliki minimal dua kolom, contoh:\n"
        "question,intent\n"
        "apakah ada obat batuk,rekomendasi_batuk\n\n"
        "Hasil pengecekan:\n"
        + "\n".join(checked_info)
    )


def load_intent_dataset() -> tuple[pd.DataFrame, str, str, Path]:
    intent_file, text_col, label_col = find_intent_dataset_file()

    print(f"\n[INFO] Intent dataset terdeteksi: {relative_to_root(intent_file)}")
    print(f"[INFO] Kolom teks  : {text_col}")
    print(f"[INFO] Kolom label : {label_col}")

    df = pd.read_csv(intent_file)
    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "intent"]

    df["text"] = df["text"].astype(str).map(clean_text)
    df["intent"] = df["intent"].astype(str).str.strip()

    df = df[
        (df["text"] != "")
        & (df["intent"] != "")
        & (df["text"].str.lower() != "nan")
        & (df["intent"].str.lower() != "nan")
    ].copy()

    df = df.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            f"Dataset intent terlalu kecil setelah cleaning: {len(df)} baris. "
            "Tambahkan minimal beberapa variasi pertanyaan per intent."
        )

    if df["intent"].nunique() < 2:
        raise ValueError(
            "Dataset intent minimal harus memiliki 2 kelas intent. "
            f"Intent terdeteksi: {df['intent'].unique().tolist()}"
        )

    return df, text_col, label_col, intent_file


def safe_train_test_split(df: pd.DataFrame):
    label_counts = df["intent"].value_counts()
    min_class_count = int(label_counts.min())
    n_classes = df["intent"].nunique()

    test_size = 0.25
    if len(df) * test_size < n_classes:
        test_size = min(0.40, max(0.25, n_classes / len(df)))

    stratify = df["intent"] if min_class_count >= 2 else None

    if stratify is None:
        print(
            "[WARN] Stratified split dinonaktifkan karena ada intent dengan jumlah data < 2. "
            "Sebaiknya tambahkan minimal 3-5 contoh pertanyaan per intent."
        )

    return train_test_split(
        df["text"],
        df["intent"],
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "multinomial_nb": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.95,
                    ),
                ),
                ("model", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


def predict_with_confidence(model: Pipeline, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(texts)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        confidence = probabilities.max(axis=1)
    else:
        confidence = np.ones(len(texts))

    return predictions, confidence


def evaluate_single_model(
    model_name: str,
    model: Pipeline,
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    print(f"\n[INFO] Training model: {model_name}")
    model.fit(x_train, y_train)

    y_pred, confidence = predict_with_confidence(model, x_test)
    labels = sorted(pd.Series(y_test).unique().tolist())

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "avg_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORTS_DIR / f"classification_report_{model_name}.csv", index=True)

    error_df = pd.DataFrame(
        {
            "text": x_test.values,
            "actual_intent": y_test.values,
            "predicted_intent": y_pred,
            "confidence": confidence,
        }
    )
    error_df["is_correct"] = error_df["actual_intent"] == error_df["predicted_intent"]
    error_df["error_type"] = np.where(
        error_df["is_correct"],
        "correct",
        error_df["actual_intent"] + "_as_" + error_df["predicted_intent"],
    )
    error_df.to_csv(REPORTS_DIR / f"error_analysis_{model_name}.csv", index=False)

    summary_error = (
        error_df[~error_df["is_correct"]]
        .groupby(["actual_intent", "predicted_intent"])
        .size()
        .reset_index(name="jumlah_error")
        .sort_values("jumlah_error", ascending=False)
    )
    summary_error.to_csv(REPORTS_DIR / f"error_summary_{model_name}.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), max(6, len(labels) * 0.6)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close(fig)

    return {"metrics": metrics, "model": model, "error_df": error_df}


def plot_intent_distribution(df: pd.DataFrame):
    intent_counts = df["intent"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(intent_counts) * 0.35)))
    intent_counts.plot(kind="barh", ax=ax)
    ax.set_title("Distribusi Intent Dataset ApotekCare")
    ax.set_xlabel("Jumlah Pertanyaan")
    ax.set_ylabel("Intent")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intent_distribution.png", dpi=160)
    plt.close(fig)


def find_product_catalog_file() -> tuple[Path | None, str | None]:
    priority_relative_paths = [
        "data/raw/product_catalog.csv",
        "data/processed/product_catalog.csv",
        "data/product_catalog.csv",
        "data/raw/apotekcare_product_catalog.csv",
        "data/processed/apotekcare_product_catalog.csv",
        "data/raw/products.csv",
        "data/processed/products.csv",
    ]

    candidates: list[Path] = []
    for rel_path in priority_relative_paths:
        path = ROOT_DIR / rel_path
        if path.exists():
            candidates.append(path)

    for path in get_all_csv_files():
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        try:
            df_sample = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        category_col = detect_column(
            df_sample,
            PRODUCT_CATEGORY_COLUMN_CANDIDATES,
            "kategori produk",
            raise_error=False,
        )
        if category_col:
            return path, category_col

    return None, None


def plot_product_distribution():
    product_file, category_col = find_product_catalog_file()

    if product_file is None or category_col is None:
        print("[WARN] Product catalog tidak ditemukan. Visualisasi kategori produk dilewati.")
        return

    product_df = pd.read_csv(product_file)
    category_counts = product_df[category_col].astype(str).value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(category_counts) * 0.35)))
    category_counts.plot(kind="barh", ax=ax)
    ax.set_title("Distribusi Kategori Produk ApotekCare")
    ax.set_xlabel("Jumlah Produk")
    ax.set_ylabel("Kategori Produk")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "product_category_distribution.png", dpi=160)
    plt.close(fig)

    print(f"[INFO] Product catalog terdeteksi: {relative_to_root(product_file)}")
    print(f"[INFO] Kolom kategori produk: {category_col}")


def plot_model_comparison(metrics_df: pd.DataFrame):
    metric_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    plot_df = metrics_df.set_index("model")[metric_cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Perbandingan Metrik Model Intent Classification")
    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_metrics_comparison.png", dpi=160)
    plt.close(fig)


def choose_best_model(results: dict[str, dict]) -> str:
    metrics_df = pd.DataFrame([item["metrics"] for item in results.values()])
    metrics_df = metrics_df.sort_values(
        by=["f1_macro", "accuracy", "f1_weighted"],
        ascending=False,
    )
    return str(metrics_df.iloc[0]["model"])


def main():
    print("=" * 80)
    print("TAHAP 02 - EVALUASI MODEL APOTEKCARE ASSISTANT")
    print("=" * 80)

    print_data_folder_diagnostics()

    df, original_text_col, original_label_col, intent_file = load_intent_dataset()
    print(f"\n[INFO] Dataset loaded: {len(df)} rows, {df['intent'].nunique()} intents")
    print("\n[INFO] Distribusi intent:")
    print(df["intent"].value_counts().to_string())

    plot_intent_distribution(df)
    plot_product_distribution()

    x_train, x_test, y_train, y_test = safe_train_test_split(df)
    models = build_models()

    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_single_model(model_name, model, x_train, x_test, y_train, y_test)

    metrics_df = pd.DataFrame([item["metrics"] for item in results.values()])
    metrics_df = metrics_df.sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)
    plot_model_comparison(metrics_df)

    best_model_name = choose_best_model(results)
    best_model = results[best_model_name]["model"]

    print(f"\n[INFO] Best model: {best_model_name}")

    best_model.fit(df["text"], df["intent"])

    best_model_path = MODELS_DIR / "apotekcare_intent_model_stage02.pkl"
    joblib.dump(best_model, best_model_path)

    metadata = {
        "best_model": best_model_name,
        "model_path": str(best_model_path.as_posix()),
        "confidence_threshold": 0.45,
        "threshold_note": (
            "Threshold awal 0.45. Nilai ini dapat dinaikkan menjadi 0.50-0.60 "
            "jika chatbot terlalu sering memaksa menjawab."
        ),
        "n_dataset": int(len(df)),
        "n_intents": int(df["intent"].nunique()),
        "intents": sorted(df["intent"].unique().tolist()),
        "detected_dataset_file": relative_to_root(intent_file),
        "detected_text_column": original_text_col,
        "detected_label_column": original_label_col,
        "evaluation_metric_priority": "f1_macro",
    }
    with open(MODELS_DIR / "model_metadata_stage02.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Ringkasan metrik:")
    print(metrics_df.to_string(index=False))

    print("\n[SUCCESS] Evaluasi selesai.")
    print(f"[SUCCESS] Model terbaik disimpan ke: {best_model_path}")
    print(f"[SUCCESS] Report disimpan ke: {REPORTS_DIR}")
    print(f"[SUCCESS] Gambar visualisasi disimpan ke: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
