"""Training baseline model intent classification ApotekCare Assistant."""

from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.preprocessing import preprocess_text


def train_and_evaluate():
    data_path = ROOT_DIR / "data" / "raw" / "intent_dataset.csv"
    processed_path = ROOT_DIR / "data" / "processed" / "intent_dataset_clean.csv"
    model_dir = ROOT_DIR / "models"
    report_dir = ROOT_DIR / "reports"
    fig_dir = report_dir / "figures"

    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(preprocess_text)
    df.to_csv(processed_path, index=False, encoding="utf-8")

    X = df["clean_text"]
    y = df["intent"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results = {}
    best_name = None
    best_model = None
    best_accuracy = -1
    best_pred = None

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        )
        results[name] = {
            "accuracy": acc,
            "classification_report": report,
        }
        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name
            best_model = model
            best_pred = y_pred

    # Simpan model terbaik
    joblib.dump(best_model, model_dir / "intent_model.pkl")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")

    # Visualisasi distribusi intent
    intent_counts = df["intent"].value_counts()
    plt.figure(figsize=(12, 6))
    intent_counts.plot(kind="bar")
    plt.title("Distribusi Data Berdasarkan Intent")
    plt.xlabel("Intent")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "intent_distribution.png", dpi=150)
    plt.close()

    # Confusion matrix model terbaik
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=np.arange(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=90)
    plt.yticks(ticks=np.arange(len(label_encoder.classes_)), labels=label_encoder.classes_)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Distribusi kategori produk
    product_df = pd.read_csv(ROOT_DIR / "data" / "raw" / "product_catalog.csv")
    product_counts = product_df["category"].value_counts()
    plt.figure(figsize=(10, 5))
    product_counts.plot(kind="bar")
    plt.title("Distribusi Kategori Produk")
    plt.xlabel("Kategori Produk")
    plt.ylabel("Jumlah Produk")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "product_category_distribution.png", dpi=150)
    plt.close()

    # Simpan laporan evaluasi
    result_payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "best_model": best_name,
        "best_accuracy": best_accuracy,
        "models": results,
    }
    (report_dir / "evaluation_metrics.json").write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Evaluation Report - ApotekCare Assistant",
        "",
        f"Generated at: {result_payload['generated_at']}",
        f"Best model: **{best_name}**",
        f"Best accuracy: **{best_accuracy:.4f}**",
        "",
        "## Model Comparison",
        "",
        "| Model | Accuracy |",
        "|---|---:|",
    ]
    for name, payload in results.items():
        lines.append(f"| {name} | {payload['accuracy']:.4f} |")
    lines.extend([
        "",
        "## Catatan Interpretasi",
        "",
        "Model baseline menggunakan TF-IDF sebagai representasi fitur teks dan supervised learning untuk klasifikasi intent.",
        "Evaluasi menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix.",
        "Dataset bersifat simulatif untuk kebutuhan akademik dan tidak digunakan sebagai dasar diagnosis medis.",
    ])
    (report_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Training selesai.")
    print(f"Best model: {best_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Model disimpan di: {model_dir}")
    print(f"Report disimpan di: {report_dir}")


if __name__ == "__main__":
    train_and_evaluate()
