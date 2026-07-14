"""Build the single Parquet article dataset used by Days 1-4."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.processing.cleaning import clean_text_pipeline, extract_article_structure


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "v1"
SOURCE_PATH = PROJECT_ROOT / "data/external/kaggle_news/final_merge_dataset.csv"
IHSG_PATH = PROJECT_ROOT / "data/external/ihsg/ihsg_jkse_2024-07-31_2025-10-23.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
ARTICLES_PATH = OUTPUT_DIR / "articles.parquet"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"
CHUNK_SIZE = 2_000
MIN_RAW_CONTENT_LENGTH = 150
MIN_CLEAN_CONTENT_LENGTH = 100
ARTICLE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("article_url", pa.string()),
        ("source", pa.string()),
        ("published_at", pa.timestamp("us")),
        ("title", pa.string()),
        ("content_clean", pa.string()),
        ("dateline_location", pa.string()),
        ("dateline_publisher", pa.string()),
        ("section_label", pa.string()),
        ("quoted_text", pa.string()),
        ("quote_count", pa.int64()),
        ("content_entity_normalized", pa.string()),
        ("entity_aliases_json", pa.string()),
        ("analysis_eligibility", pa.string()),
    ]
)

INDONESIAN_MONTHS = {
    "januari": "January", "jan": "January",
    "februari": "February", "feb": "February",
    "maret": "March", "mar": "March",
    "april": "April", "apr": "April",
    "mei": "May",
    "juni": "June", "jun": "June",
    "juli": "July", "jul": "July",
    "agustus": "August", "agu": "August", "agt": "August",
    "september": "September", "sep": "September",
    "oktober": "October", "okt": "October",
    "november": "November", "nov": "November",
    "desember": "December", "des": "December",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_indonesian_date(value: object) -> datetime | None:
    """Parse only publication-date formats observed in the local Kaggle file."""
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).lower().strip()
    cleaned = re.sub(r"\b(senin|selasa|rabu|kamis|jumat|jum'at|sabtu|minggu)\b,?\s*", "", cleaned)
    cleaned = cleaned.replace("wib", "").replace("|", " ")
    cleaned = re.sub(r"(?<=\d)\.(?=\d{2}\b)", ":", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for indonesian, english in INDONESIAN_MONTHS.items():
        cleaned = re.sub(rf"\b{indonesian}\b", english, cleaned)
    for date_format in (
        "%d %B %Y %H:%M",
        "%d %B %Y %H:%M:%S",
        "%d %B %Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, date_format)
        except ValueError:
            continue
    return None


def canonicalize_url(value: object) -> str:
    parsed = urlsplit(str(value).strip())
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, urlencode(query), ""))


def build_article_id(source: str, article_url: str, title: str, published_at: datetime) -> str:
    identity = article_url or f"{source}|{title}|{published_at.isoformat()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _process_chunk(chunk: pd.DataFrame, seen_ids: set[str], stats: dict) -> pd.DataFrame:
    records = []
    for row in chunk.itertuples(index=False):
        title = str(getattr(row, "Judul", "")).strip()
        raw_value = getattr(row, "Content", "")
        raw_content = "" if pd.isna(raw_value) else str(raw_value).strip()
        if not title or len(raw_content) < MIN_RAW_CONTENT_LENGTH:
            stats["invalid_rows"] += 1
            continue
        published_at = parse_indonesian_date(getattr(row, "Waktu", None))
        if published_at is None:
            stats["invalid_rows"] += 1
            continue
        source = str(getattr(row, "source", "")).strip().lower()
        link_value = getattr(row, "Link", "")
        article_url = "" if pd.isna(link_value) else canonicalize_url(link_value)
        article_id = build_article_id(source, article_url, title, published_at)
        if article_id in seen_ids:
            stats["exact_duplicates"] += 1
            continue
        if not article_url:
            stats["invalid_rows"] += 1
            continue
        basic_clean = clean_text_pipeline(raw_content, source=source)
        structured = extract_article_structure(basic_clean, source, precleaned=True)
        seen_ids.add(article_id)
        stats["source_distribution"][source] = stats["source_distribution"].get(source, 0) + 1
        stats["published_at_min"] = min(stats["published_at_min"], published_at) if stats["published_at_min"] else published_at
        stats["published_at_max"] = max(stats["published_at_max"], published_at) if stats["published_at_max"] else published_at
        stats["datelines_extracted"] += int(structured["dateline_location"] is not None)
        stats["sections_extracted"] += int(structured["section_label"] is not None)
        stats["quotes_extracted"] += int(structured["quote_count"])
        stats["explicit_alias_maps"] += int(structured["entity_aliases_json"] != "{}")
        records.append(
            {
                "article_id": article_id,
                "article_url": article_url,
                "source": source,
                "published_at": published_at,
                "title": title,
                "content_clean": structured["content_clean"],
                "dateline_location": structured["dateline_location"],
                "dateline_publisher": structured["dateline_publisher"],
                "section_label": structured["section_label"],
                "quoted_text": structured["quoted_text"],
                "quote_count": structured["quote_count"],
                "content_entity_normalized": (
                    structured["content_entity_normalized"]
                    if structured["entity_aliases_json"] != "{}"
                    else None
                ),
                "entity_aliases_json": structured["entity_aliases_json"],
                "analysis_eligibility": (
                    "eligible"
                    if len(structured["content_clean"]) >= MIN_CLEAN_CONTENT_LENGTH
                    else "too_short_after_metadata_removal"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_dataset() -> dict:
    """Transform immutable local inputs directly into one NLP-ready Parquet."""
    if not SOURCE_PATH.exists() or not IHSG_PATH.exists():
        raise FileNotFoundError("The local Kaggle article CSV and IHSG Parquet are required.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_articles = ARTICLES_PATH.with_suffix(".parquet.tmp")
    temporary_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_articles.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)
    stats = {
        "input_rows": 0,
        "invalid_rows": 0,
        "exact_duplicates": 0,
        "datelines_extracted": 0,
        "sections_extracted": 0,
        "quotes_extracted": 0,
        "explicit_alias_maps": 0,
        "source_distribution": {},
        "published_at_min": None,
        "published_at_max": None,
    }
    seen_ids: set[str] = set()
    writer = None
    try:
        for chunk in pd.read_csv(SOURCE_PATH, chunksize=CHUNK_SIZE):
            stats["input_rows"] += len(chunk)
            processed = _process_chunk(chunk, seen_ids, stats)
            if processed.empty:
                continue
            table = pa.Table.from_pandas(processed, schema=ARTICLE_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_articles, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Dataset build produced no valid article rows.")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pipeline": "immutable local CSV -> source-aware cleaning -> exact URL dedup -> Parquet",
        "inputs": {
            "articles": {"path": str(SOURCE_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_file(SOURCE_PATH)},
            "ihsg": {"path": str(IHSG_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_file(IHSG_PATH)},
        },
        "output": {
            "path": str(ARTICLES_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(temporary_articles),
            "row_count": len(seen_ids),
            "columns": ARTICLE_SCHEMA.names,
            "published_at_min": stats["published_at_min"].isoformat(),
            "published_at_max": stats["published_at_max"].isoformat(),
            "source_distribution": dict(sorted(stats["source_distribution"].items())),
        },
        "quality": {
            **{
                key: value
                for key, value in stats.items()
                if key not in {"source_distribution", "published_at_min", "published_at_max"}
            },
            "required_field_nulls": {
                column: 0
                for column in ("article_id", "article_url", "source", "published_at", "title", "content_clean")
            },
        },
    }
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary_articles.replace(ARTICLES_PATH)
    temporary_manifest.replace(MANIFEST_PATH)
    logger.info("Built %s clean articles at %s", len(seen_ids), ARTICLES_PATH)
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset()
