# Market Sentinel — Data Engineering & DS/ML

Pipeline batch untuk membangun early-warning dari berita Indonesia berdasarkan empat sinyal: framing/clickbait, emosi pemicu chaos, pola propaganda, dan relevansi pasar. Tahap aktif masih Data Engineering + DS/ML; API dan frontend belum menjadi dependency.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.pipeline
python3 -m unittest discover -s tests -v
```

`src.pipeline` menjalankan satu alur: verifikasi/unduh input eksternal → cleaning dan exact dedup → label package → feature frame → EDA. Bila hanya ingin menyiapkan data:

```bash
python3 -m src.download_data
```

Downloader bersifat idempotent: file yang checksum-nya benar tidak diunduh ulang, sedangkan file rusak dihentikan dengan error. Tidak ada scraping atau database yang dibutuhkan.

## Data

| Input | Penanganan |
|---|---|
| Indonesia News Dataset 2025, Kaggle versi 2 | Diunduh otomatis; 216 MB CSV, CC BY-NC 4.0 |
| `Rifky/indonesian-hoax-news`, revisi terkunci | Diunduh otomatis; 57 MB Parquet |
| Snapshot IHSG 31-07-2024–23-10-2025 | Disimpan di Git karena hanya 19 KB |
| InSet lexicon | Disimpan di Git karena kecil |

Dataset besar dan output yang dapat dibangun ulang sengaja tidak menjadi blob Git. Hasil penting yang tidak dapat direkonstruksi, seperti CSV anotasi yang sudah selesai, tetap dapat di-commit.

## Artefak utama

- `data/processed/articles.parquet`: corpus bersih/NLP-ready.
- `data/processed/features/article_features.parquet`: feature deterministik dan split temporal.
- `data/processed/eda/`: plot dan metrik deskriptif.
- `data/processed/labels/`: template dan hasil anotasi.

Jalankan perintah dari root repository. Jangan commit `.env` atau API key. Untuk melanjutkan utilitas anotasi yang sudah tersedia:

```bash
export OPENROUTER_API_KEY="..."
python3 -m src.annotate --input INPUT.csv --out OUTPUT.csv --annotator A
```
