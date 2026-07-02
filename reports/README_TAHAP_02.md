# TAHAP 02 — Evaluasi Model, Error Analysis, Optimasi Chatbot, dan Finalisasi ZIP ApotekCare Assistant

## Tujuan
Tahap ini memperkuat kualitas chatbot ApotekCare melalui evaluasi model intent classification, perbandingan Logistic Regression dan Multinomial Naive Bayes, error analysis, confidence threshold, visualisasi, dan finalisasi ZIP untuk lampiran UAS Data Mining.

## File yang ditambahkan/diperbaiki
- `scripts/evaluate_models.py`
- `app/chatbot_engine.py`
- `app/app.py`
- `scripts/create_stage02_zip.py`
- `notebooks/02_evaluasi_model_error_analysis_optimasi_chatbot_apotekcare.ipynb`
- `reports/README_TAHAP_02.md`

## Cara menjalankan
```bash
conda activate apotekcare-dm
python scripts/evaluate_models.py
streamlit run app/app.py
python scripts/create_stage02_zip.py
```

## Output evaluasi
- `reports/model_comparison_metrics.csv`
- `reports/classification_report_logistic_regression.csv`
- `reports/classification_report_multinomial_nb.csv`
- `reports/error_analysis_logistic_regression.csv`
- `reports/error_analysis_multinomial_nb.csv`
- `reports/figures/intent_distribution.png`
- `reports/figures/confusion_matrix_logistic_regression.png`
- `reports/figures/confusion_matrix_multinomial_nb.png`
- `reports/figures/model_metrics_comparison.png`
- `reports/figures/product_category_distribution.png`

## Catatan akademik
Model dievaluasi menggunakan accuracy, precision, recall, F1-score, classification report, dan confusion matrix. F1-macro digunakan sebagai metrik utama karena intent classification dapat memiliki distribusi kelas yang tidak seimbang. Confidence threshold ditambahkan agar chatbot tidak memaksa menjawab ketika prediksi model lemah.
