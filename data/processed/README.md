# Artefak Aktif Hari 1–4

| Lokasi | Isi |
|---|---|
| `articles.parquet` | Satu corpus artikel bersih/NLP-ready, 67.373 row |
| `dataset_manifest.json` | Checksum input/output dan audit filtering |
| `labels/` | Audit label, 300 sampel umum + 300 sampel kasus chaos tanpa overlap, dan finalizer output |
| `eda/` | Satu set plot dan metrik EDA |
| `features/` | Feature frame, contract, dan split temporal |

Semua artefak dibangun dari input immutable dengan `python3 -m src.pipeline`. Tidak ada snapshot canonical/preprocessed ganda dan tidak ada database mirror.
