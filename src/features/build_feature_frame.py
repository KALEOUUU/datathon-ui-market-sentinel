"""Build Day-4 leakage-safe numeric article features from the frozen corpus."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.features.feature_contract import (
    CLICKBAIT_TRIGGER_TERMS,
    CONTRACT_PATH,
    FEATURE_VERSION,
    MARKET_KEYWORDS,
    write_feature_contract,
)
from src.features.lexicon_scoring import InSetScorer, build_inset_scorer, tokenize
from src.processing.build_dataset import ARTICLES_PATH, DATASET_VERSION, MANIFEST_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data/processed/features"
FEATURE_FRAME_PATH = OUTPUT_DIR / "article_features.parquet"
SPLIT_PLAN_PATH = OUTPUT_DIR / "temporal_split_plan.json"
SENTENCE_PATTERN = re.compile(r"[.!?]+")


def _load_inset() -> tuple[dict[str, float], dict[str, float]]:
    positive = pd.read_csv(PROJECT_ROOT / "data/external/inset_lexicon/positive.tsv", sep="\t")
    negative = pd.read_csv(PROJECT_ROOT / "data/external/inset_lexicon/negative.tsv", sep="\t")
    return (
        dict(zip(positive["word"].str.lower(), positive["weight"])),
        dict(zip(negative["word"].str.lower(), negative["weight"])),
    )


def _tokens(text: str) -> list[str]:
    return tokenize(text)


def _sentiment_density(
    text: str,
    positive_or_scorer: dict[str, float] | InSetScorer,
    negative: dict[str, float] | None = None,
) -> float:
    """Score InSet terms and phrases; overlapping polarities are neutralized."""
    scorer = positive_or_scorer if isinstance(positive_or_scorer, InSetScorer) else build_inset_scorer(positive_or_scorer, negative or {})
    return scorer.density(text)


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in str(text) if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    """Count documented terms and phrases with Unicode word boundaries.

    One compiled alternation scans each article once. Longest alternatives come
    first, so a phrase is counted as one signal rather than repeatedly through
    its component words.
    """
    return len(_term_pattern(tuple(terms)).findall(str(text).lower()))


@lru_cache(maxsize=None)
def _term_pattern(terms: tuple[str, ...]) -> re.Pattern:
    normalized_terms = [" ".join(_tokens(term)) for term in terms]
    alternatives = [re.escape(term).replace(r"\ ", r"\s+") for term in normalized_terms if term]
    return re.compile(r"(?<!\w)(?:" + "|".join(sorted(alternatives, key=len, reverse=True)) + r")(?!\w)")


def _temporal_split(dates: pd.Series) -> tuple[pd.Series, dict]:
    """Assign chronological provisional splits by unique publication date, never randomly."""
    unique_dates = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values().reset_index(drop=True)
    if len(unique_dates) < 3:
        raise ValueError("At least three unique publication dates are required for a temporal split plan.")
    train_end_index = max(0, int(len(unique_dates) * 0.70) - 1)
    validation_end_index = max(train_end_index + 1, int(len(unique_dates) * 0.85) - 1)
    validation_end_index = min(validation_end_index, len(unique_dates) - 2)
    train_end = unique_dates.iloc[train_end_index]
    validation_end = unique_dates.iloc[validation_end_index]
    normalized_dates = pd.to_datetime(dates).dt.normalize()
    split = pd.Series("test", index=dates.index, dtype="string")
    split.loc[normalized_dates <= train_end] = "train"
    split.loc[(normalized_dates > train_end) & (normalized_dates <= validation_end)] = "validation"
    plan = {
        "status": "provisional_pending_label_distribution_review",
        "method": "chronological split by unique publication dates; no random sampling",
        "proportions_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "cutoffs": {
            "train_end": train_end.date().isoformat(),
            "validation_end": validation_end.date().isoformat(),
            "test_start": unique_dates.iloc[validation_end_index + 1].date().isoformat(),
        },
        "embargo": {
            "days": 0,
            "rationale": "No label-dependent rolling target is materialized in Day 4. Reassess before any future-window market target is trained.",
        },
        "fit_policy": "All learned transformations, including TF-IDF and category encoding, must fit on train only in Day 5.",
    }
    return split, plan


def build_feature_frame() -> dict:
    """Materialize deterministic article-time features without labels or future market values."""
    if not ARTICLES_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Build data/processed/articles.parquet before feature construction.")
    write_feature_contract()
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    articles = pd.read_parquet(
        ARTICLES_PATH,
        columns=["article_id", "source", "published_at", "title", "content_clean", "quote_count"],
    )
    articles["published_at"] = pd.to_datetime(articles["published_at"])
    positive, negative = _load_inset()
    inset_scorer = build_inset_scorer(positive, negative)

    features = pd.DataFrame(
        {
            "article_id": articles["article_id"],
            "source": articles["source"],
            "published_at": articles["published_at"],
            "feature_version": FEATURE_VERSION,
            "title_char_count": articles["title"].astype(str).str.len(),
            "title_token_count": articles["title"].map(lambda text: len(_tokens(text))),
            "title_exclamation_count": articles["title"].astype(str).str.count("!"),
            "title_question_count": articles["title"].astype(str).str.count(r"\?"),
            "title_uppercase_ratio": articles["title"].map(_uppercase_ratio),
            "title_digit_count": articles["title"].astype(str).str.count(r"\d"),
            "title_clickbait_trigger_count": articles["title"].map(lambda text: _term_count(text, CLICKBAIT_TRIGGER_TERMS)),
            "content_token_count": articles["content_clean"].map(lambda text: len(_tokens(text))),
            "content_sentence_proxy_count": articles["content_clean"].astype(str).map(lambda text: len(SENTENCE_PATTERN.findall(text))),
            "content_quote_count": articles["quote_count"].astype("int64"),
            "content_digit_ratio": articles["content_clean"].astype(str).map(
                lambda text: sum(char.isdigit() for char in text) / len(text) if text else 0.0
            ),
            "inset_sentiment_density": articles["content_clean"].map(lambda text: _sentiment_density(text, inset_scorer)),
            "market_keyword_count": (articles["title"].fillna("") + " " + articles["content_clean"].fillna("")).map(
                lambda text: _term_count(text, MARKET_KEYWORDS)
            ),
        }
    )
    features["inset_sentiment_extremity"] = features["inset_sentiment_density"].abs()
    features["market_keyword_present"] = (features["market_keyword_count"] > 0).astype("int8")
    features["temporal_split"] , split_plan = _temporal_split(features["published_at"])
    if features.isna().any().any() or features["article_id"].duplicated().any():
        raise ValueError("Feature frame validation failed: null values or duplicate article_id.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.sort_values(["published_at", "article_id"], kind="stable").to_parquet(FEATURE_FRAME_PATH, index=False)
    split_plan.update(
        {
            "feature_version": FEATURE_VERSION,
            "dataset_version": DATASET_VERSION,
            "articles_sha256": dataset_manifest["output"]["sha256"],
            "row_counts": features["temporal_split"].value_counts().sort_index().to_dict(),
            "feature_frame_path": str(FEATURE_FRAME_PATH.relative_to(PROJECT_ROOT)),
            "feature_contract_path": str(CONTRACT_PATH.relative_to(PROJECT_ROOT)),
            "excluded_from_feature_frame": ["ihsg_return_t1", "realized_volatility_future", "labels", "TF-IDF matrix"],
        }
    )
    with SPLIT_PLAN_PATH.open("w", encoding="utf-8") as handle:
        json.dump(split_plan, handle, ensure_ascii=False, indent=2)
    return split_plan


if __name__ == "__main__":
    build_feature_frame()
