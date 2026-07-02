# Dokumentasi Dataset ApotekCare Assistant

Dataset pada folder ini bersifat simulatif untuk kebutuhan akademik UAS Data Mining.

## 1. faq_apotekcare.csv

Dataset FAQ berisi pasangan pertanyaan dan jawaban layanan apotek.

Kolom:
- `faq_id`: kode FAQ.
- `question`: pertanyaan umum pelanggan.
- `answer`: jawaban chatbot.
- `intent`: kategori pertanyaan.
- `safety_note`: catatan keamanan atau batasan jawaban.

## 2. intent_dataset.csv

Dataset supervised learning untuk klasifikasi intent.

Kolom:
- `text`: variasi pertanyaan pelanggan.
- `intent`: label target klasifikasi.

## 3. product_catalog.csv

Dataset katalog produk/kategori produk untuk rekomendasi sederhana berbasis intent.

Kolom:
- `product_id`: kode produk.
- `product_name`: nama produk/kategori produk.
- `category`: kategori produk.
- `related_intent`: intent yang berhubungan dengan produk.
- `description`: deskripsi produk.
- `recommendation_rule`: aturan sederhana penampilan rekomendasi.
- `safety_note`: catatan keamanan.

## 4. out_of_scope_questions.csv

Dataset pengujian pertanyaan di luar konteks dan batasan medis.

Kolom:
- `question`: pertanyaan uji.
- `expected_response`: respons yang diharapkan.
- `label`: kategori out-of-scope atau medical boundary.
