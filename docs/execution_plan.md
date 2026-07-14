# Execution Plan — Data Engineering + DS/ML (10 Step)

> **Status 14 Juli 2026:** Step 1–4 selesai sebagai pipeline Parquet-only. Input besar dapat dipulihkan oleh bootstrap ber-checksum; ini bukan live ingestion. Step 5 belum dimulai karena label belum difinalisasi. API, frontend, dashboard, PostgreSQL, dan data live ditunda.

## Tujuan

Membangun early-warning berbasis artikel Indonesia untuk empat sinyal yang dapat dijelaskan:

1. framing/clickbait;
2. emosi ekstrem yang berpotensi memicu chaos;
3. kemiripan pola propaganda/hoaks;
4. relevansi terhadap pasar Indonesia.

IHSG adalah konteks dampak finansial setelah skor berita tersedia. Korelasi EDA bukan bukti kausal.

## Pipeline aktif

```text
Kaggle v2 + SHA-256 ──> cleaning + exact URL dedup ──> articles.parquet
                                                        ├──> labels/
                                                        ├──> eda/
                                                        └──> features/
IHSG Parquet immutable ─────────────────────────────────────> eda/
InSet immutable ────────────────────────────────────────────> eda/ + features/
Hoax-news pinned + SHA-256 ─────────────────────────────────> referensi Step 7
```

Semua input dan artefak Step 1–4 dapat dipulihkan/dibangun ulang dengan:

```bash
python3 -m src.pipeline
```

## Step 1 — Audit Input dan Scope — Selesai

| Hasil | Validasi |
|---|---|
| Source artikel | Hanya `data/external/kaggle_news/final_merge_dataset.csv` |
| Source pasar | Hanya snapshot IHSG lokal, 295 hari bursa |
| Source referensi | Indonesian hoax news 17.806 judul dan InSet lokal |
| Scope | Tidak ada scraping/live ingestion, RSS, scheduler, database, API, atau frontend; hanya bootstrap input publik terkunci |
| Dependency | Hanya library pembacaan Parquet/CSV, cleaning, feature, dan visualisasi |

**Gate Step 1:** seluruh input sudah lokal dan immutable. **Lulus.**

## Step 2 — Dataset Bersih dan EDA — Selesai

| Hasil | Validasi |
|---|---|
| Build artikel | 80.472 input → 16 invalid → 13.083 exact duplicate → **67.373 artikel** |
| Identitas | UUID v5 deterministik dari URL kanonis; `article_id` dan URL unik |
| Cleaning | HTML/boilerplate dibersihkan, dateline/publisher dipisah, section Tempo memakai allowlist, kutipan dan alias eksplisit dipisah |
| Artikel pendek | 7 baris dipertahankan dengan flag `too_short_after_metadata_removal` |
| Storage | Satu `data/processed/articles.parquet` + checksum manifest |
| EDA | Satu folder `data/processed/eda/`; membaca artikel dan IHSG langsung dari Parquet |

**Gate Step 2:** dataset traceable, tanpa metadata terpotong dan tanpa state eksternal. **Lulus.**

## Step 3 — Label Governance — Selesai untuk Kode, Menunggu Manusia

| Hasil | Validasi |
|---|---|
| Strategi | Single annotator untuk kecepatan datathon; status hasil `exploratory_single_annotator` |
| Sampel umum | 300 artikel, stratified secara deterministik menurut source dan bulan |
| Sampel kasus chaos | 300 artikel pada 25–31 Agustus 2025, stratified menurut tanggal dan source; tidak overlap dengan sampel umum |
| Label | `clickbait_framing`, `chaos_prone_emotion`, `propaganda_pattern`, `market_relevance` |
| Nilai | `yes`, `no`, atau `uncertain`; rationale opsional |
| Finalizer | Memeriksa satu atau kedua paket (maksimal 600 ID unik), satu annotator, status selesai, dan nilai label sebelum membuat label Parquet |

**Gate Step 3:** workflow kode lulus. Training tetap menunggu CSV yang sudah diisi dan difinalisasi.

## Step 4 — EDA DS/ML, Features, dan Split — Selesai

| Hasil | Validasi |
|---|---|
| EDA tunggal | Source, panjang teks, coverage waktu, InSet, event 25–31 Agustus 2025, dan konteks IHSG |
| InSet | Frasa dicocokkan longest-first; term yang ada di dua polaritas dikeluarkan |
| Market keyword | Frasa seperti `bank indonesia` dan `suku bunga` dihitung benar |
| Feature frame | 67.373 row, ID sama dengan artikel, tanpa null |
| Leakage | IHSG masa depan dan label tidak masuk feature input |
| Split | Kronologis 70/15/15; transformer terpelajar baru boleh fit pada train di Step 5 |

**Gate Step 4:** feature contract dan split lulus. **Stop di sini sampai labeling selesai.**

## Step 5 — Baseline Cepat

- Finalisasi CSV single annotator.
- TF-IDF title+content yang fit pada train saja.
- Logistic Regression per task sebagai baseline utama; bandingkan majority baseline.
- Laporkan F1, precision, recall, confusion matrix, dan error penting.

**Gate:** hanya task dengan kelas `yes` dan `no` boleh dilatih.

## Step 6 — Kandidat Model dan Error Analysis

- Coba LightGBM/XGBoost hanya bila baseline perlu peningkatan.
- Review false positive/negative dan class imbalance.
- IndoBERT hanya jika error analysis menunjukkan konteks tidak tertangkap baseline dan waktu memungkinkan.

## Step 7 — Propaganda Similarity

- Gunakan hoax-news sebagai corpus referensi, bukan label artikel.
- Pastikan query dan corpus memakai encoder yang sama; abaikan embedding bawaan bila asal encoder tidak terbukti.
- Simpan top-k evidence agar similarity dapat dijelaskan.

## Step 8 — Composite Risk Score

- Gabungkan clickbait, chaos/emotion, propaganda similarity, dan market relevance.
- Bobot ditentukan dari validation atau kebijakan eksplisit yang dilaporkan; tidak membuat angka tanpa dasar.
- Simpan batch score dan evidence per artikel.

## Step 9 — Analisis Market

- Agregasikan score berita per hari tanpa leakage.
- Uji ADF, Granger/CCF, dan event study bila jumlah observasi memadai.
- Laporkan hasil negatif; jangan mengubah hubungan prediktif menjadi klaim kausal.

## Step 10 — Reproducibility dan Report

- Jalankan ulang pipeline, test, training, scoring, dan market analysis.
- Finalisasi technical report, error analysis, explainability, limitasi single annotator, dan materi presentasi.
- Product/API baru boleh dipertimbangkan setelah model selesai.

## Guardrails

- Tidak ada pseudo-label dari InSet, keyword, atau corpus hoaks.
- Tidak ada random split untuk evaluasi utama.
- Tidak ada IHSG t+1 sebagai input model artikel.
- Tidak ada komponen infra baru sebelum terbukti diperlukan.
