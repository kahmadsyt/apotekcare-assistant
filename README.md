# ApotekCare Assistant

**Judul project:** Pengembangan Chatbot Layanan Apotek Berbasis Natural Language Processing dan Text Mining untuk Klasifikasi Pertanyaan Pelanggan dan Rekomendasi Produk.

**Nama aplikasi:** ApotekCare Assistant

Project ini dibuat untuk UAS Mata Kuliah Data Mining S2 Magister Teknik Informatika UNPAM.

## Batasan Sistem

ApotekCare Assistant tidak melakukan diagnosis medis, tidak membaca resep dokter menggunakan OCR, dan tidak memberikan rekomendasi obat keras atau antibiotik secara otomatis. Untuk kondisi khusus, obat resep, ibu hamil, bayi, atau gejala berat, pengguna diarahkan berkonsultasi dengan dokter atau apoteker.

## Cara Menjalankan di Lokal Windows

Buka Anaconda Prompt atau Command Prompt, lalu masuk ke folder project:

```bash
cd apotekcare-assistant
```

Buat environment Python, lalu install dependency:

```bash
python -m venv .venv
.venv\Scriptsctivate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Latih model baseline:

```bash
python scripts/train_model.py
```

Jalankan dashboard Streamlit:

```bash
streamlit run app/app.py
```

## Cara Cepat dengan File BAT

Di Windows, Anda juga dapat menjalankan:

```bash
run_app.bat
```

File tersebut akan membuat virtual environment, install dependency, training model, dan menjalankan Streamlit.

## Struktur Folder

```text
apotekcare-assistant/
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── requirements.txt
└── README.md
```

## Dataset Utama

- `data/raw/faq_apotekcare.csv`
- `data/raw/intent_dataset.csv`
- `data/raw/product_catalog.csv`
- `data/raw/out_of_scope_questions.csv`

```
