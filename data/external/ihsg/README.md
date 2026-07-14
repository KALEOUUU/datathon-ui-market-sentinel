# Snapshot IHSG

`ihsg_jkse_2024-07-31_2025-10-23.parquet` adalah snapshot lokal data harian indeks `^JKSE` yang dipakai pipeline pada rentang corpus berita kanonis.

| Properti | Nilai |
|---|---|
| Sumber | Yahoo Finance melalui `yfinance` (`^JKSE`) |
| Rentang | 31 Juli 2024–23 Oktober 2025 |
| Jumlah | 295 hari bursa |
| Kolom | `date`, `open_price`, `high_price`, `low_price`, `close_price`, `volume`, `daily_return` |
| Pemakaian | Dibaca langsung oleh `python3 -m src.processing.eda` |

Snapshot ini adalah input immutable. Pipeline tidak mengunduh ulang, menimpa, atau menyalinnya ke database. `daily_return` sudah dihitung memakai hari bursa sebelumnya; return hari pertama rentang tidak dipaksa menjadi nol.

Snapshot ini adalah artefak reproducibility, bukan data live. Jangan memperluas rentang atau mengunduh ulang tanpa memperbarui manifest/plan eksperimen yang terkait.
