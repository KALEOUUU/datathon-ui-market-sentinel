"""Audit label availability and prepare a reproducible manual-annotation package."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.processing.build_dataset import ARTICLES_PATH, DATASET_VERSION, MANIFEST_PATH, sha256_file


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE_DIR = PROJECT_ROOT / "data/processed/labels"
AUDIT_PATH = GOVERNANCE_DIR / "label_availability_audit.json"
ANNOTATION_TEMPLATE_PATH = GOVERNANCE_DIR / "annotation_template.csv"
SAMPLE_MANIFEST_PATH = GOVERNANCE_DIR / "annotation_sample_manifest.json"
CHAOS_TEMPLATE_PATH = GOVERNANCE_DIR / "annotation_template_chaos_2025-08-25_2025-08-31.csv"
CHAOS_SAMPLE_MANIFEST_PATH = GOVERNANCE_DIR / "annotation_sample_chaos_2025-08-25_2025-08-31_manifest.json"
FINALIZED_LABELS_PATH = GOVERNANCE_DIR / "finalized_single_annotator_labels.parquet"
FINALIZATION_MANIFEST_PATH = GOVERNANCE_DIR / "single_annotator_finalization_manifest.json"
ANNOTATION_SAMPLE_SIZE = 300
ANNOTATION_SEED = "underdogs-day3-v1"
CHAOS_SAMPLE_SIZE = 300
CHAOS_SAMPLE_SEED = "underdogs-chaos-2025-08-25_2025-08-31-v1"
CHAOS_START = pd.Timestamp("2025-08-25")
CHAOS_END_EXCLUSIVE = pd.Timestamp("2025-09-01")
CHAOS_WINDOW_EVIDENCE = [
    {
        "source": "Joint statement of Indonesia's national human-rights institutions",
        "url": "https://komnasperempuan.go.id/siaran-pers-detail/siaran-pers-sikap-lembaga-nasional-hak-asasi-manusia-terhadap-aksi-demonstrasi-di-berbagai-daerah-di-indonesia-dan-penanganannya",
        "supports": "The monitored nationwide action period is stated as 25--31 August 2025.",
    },
    {
        "source": "Associated Press",
        "url": "https://apnews.com/article/indonesia-student-protest-parliament-49e31c7074aab8375aec06143f6b2edc",
        "supports": "Protests and clashes at parliament were reported on 25 August 2025.",
    },
]

LABEL_FIELDS = {
    "clickbait_framing": ["yes", "no", "uncertain"],
    "chaos_prone_emotion": ["yes", "no", "uncertain"],
    "propaganda_pattern": ["yes", "no", "uncertain"],
    "market_relevance": ["yes", "no", "uncertain"],
}


def _stable_rank(article_id: str, seed: str = ANNOTATION_SEED) -> str:
    return hashlib.sha256(f"{seed}|{article_id}".encode("utf-8")).hexdigest()


def _proportional_allocation(group_sizes: pd.Series, sample_size: int) -> pd.Series:
    """Allocate a fixed review budget proportionally, preserving every nonempty stratum when possible."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= int(group_sizes.sum()):
        return group_sizes.astype(int)

    ideal = group_sizes / group_sizes.sum() * sample_size
    allocation = ideal.astype(int)
    if sample_size >= len(group_sizes):
        allocation = allocation.clip(lower=1)
    while allocation.sum() > sample_size:
        candidates = allocation[allocation > 1]
        allocation.loc[candidates.sort_values(ascending=False).index[0]] -= 1
    remaining = sample_size - int(allocation.sum())
    for index in (ideal - allocation).sort_values(ascending=False).index[:remaining]:
        allocation.loc[index] += 1
    return allocation.astype(int)


def _stratified_annotation_sample(articles: pd.DataFrame, sample_size: int) -> tuple[pd.DataFrame, dict]:
    working = articles.copy()
    working["annotation_month"] = pd.to_datetime(working["published_at"]).dt.to_period("M").astype(str)
    group_sizes = working.groupby(["source", "annotation_month"], observed=True).size()
    allocation = _proportional_allocation(group_sizes, min(sample_size, len(working)))
    selected = []
    for (source, month), count in allocation.items():
        group = working[(working["source"] == source) & (working["annotation_month"] == month)].copy()
        group["_rank"] = group["article_id"].map(_stable_rank)
        selected.append(group.sort_values("_rank", kind="stable").head(count))
    sample = pd.concat(selected, ignore_index=True).sort_values(["published_at", "article_id"], kind="stable")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "sampling_method": "proportional stratification by source and publication month with deterministic SHA-256 ranking",
        "seed": ANNOTATION_SEED,
        "requested_sample_size": sample_size,
        "actual_sample_size": len(sample),
        "stratum_count": len(group_sizes),
        "stratum_allocation": {f"{source}|{month}": int(count) for (source, month), count in allocation.items()},
    }
    return sample, manifest


def _chaos_annotation_sample(
    articles: pd.DataFrame,
    excluded_article_ids: set[str],
    sample_size: int = CHAOS_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, dict]:
    """Sample the verified 25--31 August 2025 event window without reusing general-sample rows."""
    working = articles.copy()
    working["published_at"] = pd.to_datetime(working["published_at"])
    in_window = working[
        (working["published_at"] >= CHAOS_START)
        & (working["published_at"] < CHAOS_END_EXCLUSIVE)
    ].copy()
    candidates_before_exclusion = len(in_window)
    working = in_window[~in_window["article_id"].isin(excluded_article_ids)].copy()
    if len(working) < sample_size:
        raise ValueError(
            "Not enough non-overlapping articles in the 25--31 August 2025 event window: "
            f"available={len(working)}, requested={sample_size}."
        )

    working["annotation_date"] = working["published_at"].dt.strftime("%Y-%m-%d")
    group_sizes = working.groupby(["annotation_date", "source"], observed=True).size()
    allocation = _proportional_allocation(group_sizes, sample_size)
    selected = []
    for (publication_date, source), count in allocation.items():
        group = working[
            (working["annotation_date"] == publication_date)
            & (working["source"] == source)
        ].copy()
        group["_rank"] = group["article_id"].map(lambda article_id: _stable_rank(article_id, CHAOS_SAMPLE_SEED))
        selected.append(group.sort_values("_rank", kind="stable").head(count))

    sample = pd.concat(selected, ignore_index=True).sort_values(["published_at", "article_id"], kind="stable")
    manifest = {
        "dataset_version": DATASET_VERSION,
        "event_window_start": CHAOS_START.date().isoformat(),
        "event_window_end_inclusive": (CHAOS_END_EXCLUSIVE - pd.Timedelta(days=1)).date().isoformat(),
        "event_window_evidence": CHAOS_WINDOW_EVIDENCE,
        "sampling_method": "proportional stratification by publication date and source with deterministic SHA-256 ranking",
        "seed": CHAOS_SAMPLE_SEED,
        "requested_sample_size": sample_size,
        "actual_sample_size": len(sample),
        "candidate_count_before_general_sample_exclusion": candidates_before_exclusion,
        "excluded_general_sample_count": candidates_before_exclusion - len(working),
        "candidate_count_after_general_sample_exclusion": len(working),
        "stratum_count": len(group_sizes),
        "stratum_allocation": {
            f"{publication_date}|{source}": int(count)
            for (publication_date, source), count in allocation.items()
        },
    }
    return sample, manifest


def _blank_template(sample: pd.DataFrame) -> pd.DataFrame:
    template = sample[["article_id", "source", "published_at", "title", "content_clean"]].copy()
    for task in LABEL_FIELDS:
        template[f"{task}_label"] = ""
    template["annotator_id"] = ""
    template["annotation_status"] = "pending"
    template["rationale"] = ""
    return template


def _write_or_preserve_template(template: pd.DataFrame, path: Path) -> str:
    """Never overwrite annotation work; reject a stale package instead."""
    if not path.exists():
        template.to_csv(path, index=False)
        return "created"
    existing = pd.read_csv(path, usecols=["article_id"], keep_default_na=False)
    if existing["article_id"].duplicated().any() or set(existing["article_id"]) != set(template["article_id"]):
        raise ValueError(f"Existing annotation package does not match the deterministic sample: {path}")
    logger.info("Preserved existing annotation template at %s; it may contain human work.", path)
    return "preserved"


def _expected_annotation_packages(articles: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict]]:
    general_sample, general_manifest = _stratified_annotation_sample(articles, ANNOTATION_SAMPLE_SIZE)
    chaos_sample, chaos_manifest = _chaos_annotation_sample(
        articles,
        set(general_sample["article_id"]),
        CHAOS_SAMPLE_SIZE,
    )
    return {
        "representative": (general_sample, general_manifest),
        "chaos_2025-08-25_2025-08-31": (chaos_sample, chaos_manifest),
    }


def audit_and_prepare_annotation_package(sample_size: int = ANNOTATION_SAMPLE_SIZE) -> dict:
    """Create an auditable label-readiness report and blank single-annotation template.

    The template deliberately contains no inferred labels. A human annotator must
    fill it according to ``docs/labeling_guidelines.md``.
    """
    if not ARTICLES_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Build data/processed/articles.parquet before label governance.")
    articles = pd.read_parquet(ARTICLES_PATH, columns=["article_id", "source", "published_at", "title", "content_clean"])
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    corpus_columns = dataset_manifest["output"]["columns"]
    existing_label_columns = sorted(set(corpus_columns).intersection(LABEL_FIELDS))

    audit = {
        "dataset_version": DATASET_VERSION,
        "corpus": {
            "path": str(ARTICLES_PATH.relative_to(PROJECT_ROOT)),
            "row_count": len(articles),
            "columns": corpus_columns,
        },
        "task_label_readiness": {
            task: {
                "ground_truth_column_present": task in existing_label_columns,
                "status": "manual_annotation_required",
                "reason": "The historical-news corpus has no verified ground-truth column for this task.",
                "allowed_values": allowed_values,
            }
            for task, allowed_values in LABEL_FIELDS.items()
        },
        "reference_corpus_policy": {
            "path": "data/external/indonesian_hoax_news/",
            "role": "propaganda/hoax similarity reference only",
            "not_used_as": "ground-truth labels for the canonical news corpus",
            "embedding_status": "not used until source encoder compatibility is validated",
        },
        "annotation_mode": "single_annotator_exploratory",
        "training_readiness": "awaiting_single_annotator_finalization",
        "prohibited": [
            "Do not treat lexicon matches as ground truth.",
            "Do not treat the hoax reference corpus as labels for unrelated news articles.",
            "Do not train a supervised classifier before the single-annotator file is validated and finalized.",
        ],
    }

    sample, sample_manifest = _stratified_annotation_sample(articles, sample_size)
    chaos_sample, chaos_sample_manifest = _chaos_annotation_sample(
        articles,
        set(sample["article_id"]),
        CHAOS_SAMPLE_SIZE,
    )
    if set(sample["article_id"]).intersection(chaos_sample["article_id"]):
        raise ValueError("General and event annotation packages must not overlap.")
    template = _blank_template(sample)
    chaos_template = _blank_template(chaos_sample)
    audit["annotation_packages"] = {
        "representative": {
            "path": str(ANNOTATION_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
            "row_count": len(template),
        },
        "chaos_2025-08-25_2025-08-31": {
            "path": str(CHAOS_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
            "row_count": len(chaos_template),
            "event_window_start": "2025-08-25",
            "event_window_end_inclusive": "2025-08-31",
        },
        "combined_unique_article_count": len(template) + len(chaos_template),
    }

    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    template_status = _write_or_preserve_template(template, ANNOTATION_TEMPLATE_PATH)
    chaos_template_status = _write_or_preserve_template(chaos_template, CHAOS_TEMPLATE_PATH)
    with SAMPLE_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(sample_manifest, handle, ensure_ascii=False, indent=2)
    with CHAOS_SAMPLE_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(chaos_sample_manifest, handle, ensure_ascii=False, indent=2)
    logger.info(
        "Label audit complete; representative template %s and chaos-window template %s.",
        template_status,
        chaos_template_status,
    )
    return {
        "audit": audit,
        "sample_manifest": sample_manifest,
        "chaos_sample_manifest": chaos_sample_manifest,
    }


def finalize_single_annotator_labels(annotation_paths: Path | str | list[Path | str]) -> dict:
    """Validate and version one or both completed single-annotator packages.

    This intentionally does not claim inter-annotator agreement. ``uncertain``
    values remain in the audited label table but are excluded from each task's
    binary training subset.
    """
    if isinstance(annotation_paths, (str, Path)):
        annotation_paths = [annotation_paths]
    paths = [Path(path) for path in annotation_paths]
    if not paths:
        raise ValueError("Provide at least one completed annotation CSV.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Completed annotation file not found: {path}")
    if FINALIZED_LABELS_PATH.exists() or FINALIZATION_MANIFEST_PATH.exists():
        raise FileExistsError(
            f"Single-annotator labels already finalized at {GOVERNANCE_DIR}; preserve them before starting a correction."
        )
    articles = pd.read_parquet(ARTICLES_PATH, columns=["article_id", "source", "published_at", "title", "content_clean"])
    expected_packages = _expected_annotation_packages(articles)
    required_columns = {
        "article_id", "annotator_id", "annotation_status", "rationale",
        *(f"{task}_label" for task in LABEL_FIELDS),
    }
    annotation_frames = []
    matched_package_names = []
    input_metadata = []
    for path in paths:
        frame = pd.read_csv(path, keep_default_na=False)
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Annotation file {path} is missing required columns: {sorted(missing_columns)}")
        if frame["article_id"].duplicated().any():
            raise ValueError(f"Annotation file contains duplicate article_id values: {path}")
        actual_ids = set(frame["article_id"])
        matches = [
            name
            for name, (sample, _) in expected_packages.items()
            if actual_ids == set(sample["article_id"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Annotation file does not exactly match a deterministic package: {path}")
        package_name = matches[0]
        if package_name in matched_package_names:
            raise ValueError(f"Annotation package supplied more than once: {package_name}")
        matched_package_names.append(package_name)
        annotation_frames.append(frame)
        input_metadata.append({"path": str(path), "sha256": sha256_file(path), "package": package_name})

    annotation = pd.concat(annotation_frames, ignore_index=True)
    if annotation["article_id"].duplicated().any():
        raise ValueError("Completed annotation packages contain overlapping article_id values.")
    annotator_ids = annotation["annotator_id"].astype(str).str.strip()
    if not annotator_ids.ne("").all() or annotator_ids.nunique() != 1:
        raise ValueError("Single-annotator finalization requires one non-empty annotator_id for every row.")
    statuses = annotation["annotation_status"].astype(str).str.strip().str.lower()
    invalid_statuses = sorted(set(statuses).difference({"completed", "done"}))
    if invalid_statuses:
        raise ValueError(
            "Every annotation_status must be 'completed' or 'done' before finalization; "
            f"invalid={invalid_statuses}."
        )
    annotation["annotation_status"] = "completed"
    for task, allowed_values in LABEL_FIELDS.items():
        column = f"{task}_label"
        annotation[column] = annotation[column].astype(str).str.strip().str.lower()
        invalid_values = sorted(set(annotation[column]).difference(allowed_values))
        if invalid_values:
            raise ValueError(f"Invalid values in {column}: {invalid_values}; allowed={allowed_values}")

    corpus_index = articles.set_index("article_id")
    expected_order = pd.concat(
        [expected_packages[name][0][["article_id"]] for name in expected_packages if name in matched_package_names],
        ignore_index=True,
    )["article_id"]
    annotation = annotation.set_index("article_id").loc[expected_order].reset_index()
    labels = annotation[["article_id", "annotator_id", "annotation_status", "rationale"]].copy()
    labels["source"] = labels["article_id"].map(corpus_index["source"])
    labels["published_at"] = labels["article_id"].map(corpus_index["published_at"])
    task_summary = {}
    for task in LABEL_FIELDS:
        raw_column = f"{task}_label"
        binary_column = f"{task}_binary"
        labels[raw_column] = annotation[raw_column]
        labels[binary_column] = annotation[raw_column].map({"yes": 1, "no": 0}).astype("Int8")
        counts = annotation[raw_column].value_counts().reindex(LABEL_FIELDS[task], fill_value=0)
        task_summary[task] = {
            "raw_label_counts": {label: int(count) for label, count in counts.items()},
            "binary_training_row_count": int(labels[binary_column].notna().sum()),
            "binary_classes_present": sorted(int(value) for value in labels[binary_column].dropna().unique()),
            "day5_readiness": (
                "ready_exploratory_single_annotator"
                if set(labels[binary_column].dropna().unique()) == {0, 1}
                else "blocked_missing_yes_or_no_class"
            ),
        }

    GOVERNANCE_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(FINALIZED_LABELS_PATH, index=False)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "label_governance_mode": "exploratory_single_annotator",
        "annotator_id": annotator_ids.iloc[0],
        "annotation_inputs": input_metadata,
        "deterministic_samples": {
            name: expected_packages[name][1]
            for name in expected_packages
            if name in matched_package_names
        },
        "finalized_labels": {
            "path": str(FINALIZED_LABELS_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(FINALIZED_LABELS_PATH),
            "row_count": len(labels),
        },
        "task_summary": task_summary,
        "training_use_policy": [
            "This is an exploratory single-annotator label set; no IAA was calculated and rationale is optional.",
            "Use only yes/no rows for the corresponding binary task; preserve uncertain rows for audit.",
            "Do not report these labels as independently validated ground truth.",
        ],
    }
    with FINALIZATION_MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    logger.info("Finalized %s single-annotator labels at %s", len(labels), FINALIZED_LABELS_PATH)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare or finalize the Day-3 single-annotator label package.")
    parser.add_argument(
        "--finalize-single",
        type=Path,
        nargs="+",
        help="One or both completed CSV packages to validate and finalize together.",
    )
    args = parser.parse_args()
    if args.finalize_single:
        finalize_single_annotator_labels(args.finalize_single)
    else:
        audit_and_prepare_annotation_package()
