# Repository Structure — STEP 1–4

```text
datathon_ui/
├── data/
│   ├── external/                         # input immutable
│   │   ├── kaggle_news/final_merge_dataset.csv
│   │   ├── ihsg/*.parquet
│   │   ├── inset_lexicon/*.tsv
│   │   └── indonesian_hoax_news/data/*.parquet
│   └── processed/
│       ├── articles.parquet
│       ├── dataset_manifest.json
│       ├── labels/
│       ├── eda/
│       └── features/
├── src/
│   ├── download_data.py
│   ├── pipeline.py
│   ├── processing/
│   │   ├── build_dataset.py
│   │   ├── cleaning.py
│   │   ├── label_governance.py
│   │   └── eda.py
│   └── features/
│       ├── lexicon_scoring.py
│       ├── feature_contract.py
│       └── build_feature_frame.py
├── tests/
├── docs/
├── requirements.txt
└── {Arsitektur Datathon Underdogs,Technical_Architecture_Plan_Underdogs}.docx
```
