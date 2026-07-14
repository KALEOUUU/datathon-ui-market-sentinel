# Project Context — Underdogs

## Problem

Project membangun sistem peringatan dini untuk menilai risiko pola informasi pada berita Indonesia: framing/clickbait, emosi ekstrem pemicu chaos, pola propaganda, dan relevansi pasar. Output akhir yang dituju adalah risk score per artikel dengan alasan yang dapat diperiksa.

IHSG digunakan untuk menguji apakah agregat sinyal berita mendahului perubahan pasar secara statistik. Ia bukan target utama, bukan label untuk artikel, dan korelasi tidak boleh disebut kausal.

## Scope aktif

- Data historis lokal saja: Detik, Kompas, dan Tempo.
- Satu pipeline batch Parquet-only.
- Single annotator untuk label eksploratif datathon.
- Baseline interpretable lebih dahulu; model kompleks hanya berdasarkan error analysis.
- Tidak ada scraping/live ingestion, PostgreSQL, Docker service, API, frontend, dashboard, scheduler, atau vector database pada Step 1–4. Downloader hanya memulihkan snapshot publik terkunci dan memverifikasi checksum.

Keputusan ini menyederhanakan storage yang lebih luas dalam dokumen arsitektur asli. PostgreSQL tetap mungkin dipakai kelak untuk serving, tetapi tidak menjadi dependency DS/ML offline.

## Dataset

| Dataset | Lokasi | Peran |
|---|---|---|
| Historical news | `data/external/kaggle_news/final_merge_dataset.csv` | Corpus artikel utama |
| IHSG | `data/external/ihsg/ihsg_jkse_2024-07-31_2025-10-23.parquet` | Konteks market 295 hari |
| InSet | `data/external/inset_lexicon/` | Feature baseline sentimen, bukan label |
| Indonesian hoax news | `data/external/indonesian_hoax_news/data/*.parquet` | Referensi similarity propaganda, bukan label artikel |

MaFindo tidak digunakan.

## Stage aktif

1. `download_data.py`: memulihkan dua input besar yang tidak disimpan di Git dan memverifikasi seluruh input eksternal.
2. `build_dataset.py`: validasi raw, parsing waktu, URL canonicalization, exact dedup, source-aware cleaning, lalu menulis satu `articles.parquet`.
3. `label_governance.py`: membuat 300 sampel umum + 300 sampel kasus chaos 25–31 Agustus 2025 tanpa overlap, lalu memfinalisasi hasil single annotator.
4. `eda.py`: satu EDA untuk kualitas corpus, readiness DS/ML, baseline InSet, dan konteks IHSG.
5. `build_feature_frame.py`: feature deterministik dan split temporal tanpa leakage.

## Success criteria

- Pipeline dapat dijalankan tanpa service eksternal.
- Dataset dan feature memiliki checksum, ID unik, serta jumlah row konsisten.
- Model Step 5 dievaluasi pada split temporal dan status label single annotator dinyatakan jujur.
- Explainability dan bukti retrieval lebih penting daripada ensemble yang rumit.
- Klaim market harus disertai metode statistik dan limitasi.

## Deferred

API, dashboard, storage serving, live ingestion, tuning besar, transformer end-to-end, dan product work ditunda sampai model serta analisis market selesai.
