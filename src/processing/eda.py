"""One Parquet-only EDA for dataset quality and Day-4 DS/ML readiness."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/underdogs-matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns

from src.features.lexicon_scoring import build_inset_scorer
from src.features.build_feature_frame import FEATURE_FRAME_PATH
from src.processing.build_dataset import ARTICLES_PATH, DATASET_VERSION, IHSG_PATH, MANIFEST_PATH


logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "data/processed/eda"
METRICS_PATH = OUTPUT_DIR / "eda_metrics.json"
EVENT_START = pd.Timestamp("2025-08-25")
EVENT_END = pd.Timestamp("2025-08-31")
SOURCE_COLORS = {"kompas": "#2563EB", "tempo": "#F97316", "detik": "#16A34A"}

sns.set_theme(style="whitegrid", context="notebook")


def _load_inset():
    positive = pd.read_csv(PROJECT_ROOT / "data/external/inset_lexicon/positive.tsv", sep="\t")
    negative = pd.read_csv(PROJECT_ROOT / "data/external/inset_lexicon/negative.tsv", sep="\t")
    return build_inset_scorer(
        dict(zip(positive["word"].str.lower(), positive["weight"])),
        dict(zip(negative["word"].str.lower(), negative["weight"])),
    )


def _save(fig, filename: str) -> None:
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _total_variation(first: pd.Series, second: pd.Series) -> float:
    index = first.index.union(second.index)
    return float(0.5 * (first.reindex(index, fill_value=0) - second.reindex(index, fill_value=0)).abs().sum())


def _equal_complete_month_windows(articles: pd.DataFrame) -> tuple[dict, pd.Series, pd.Series]:
    start = articles["published_at"].min().normalize()
    end = articles["published_at"].max().normalize()
    complete_months = pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")
    if start.day != 1:
        complete_months = complete_months[1:]
    if end.day != end.days_in_month:
        complete_months = complete_months[:-1]
    if len(complete_months) < 2:
        raise ValueError("EDA requires at least two complete calendar months for source-mix comparison.")
    count = min(3, len(complete_months) // 2)
    first_window = complete_months[:count]
    last_window = complete_months[-count:]
    article_month = articles["published_at"].dt.to_period("M")
    first_mix = articles.loc[article_month.isin(first_window), "source"].value_counts(normalize=True)
    last_mix = articles.loc[article_month.isin(last_window), "source"].value_counts(normalize=True)
    details = {
        "window_month_count": count,
        "first_window": [str(period) for period in first_window],
        "last_window": [str(period) for period in last_window],
        "first_window_row_count": int(article_month.isin(first_window).sum()),
        "last_window_row_count": int(article_month.isin(last_window).sum()),
        "total_variation": _total_variation(first_mix, last_mix),
    }
    return details, first_mix, last_mix


def run_eda() -> dict:
    """Generate the only EDA artifact set used by Days 1-4."""
    if not ARTICLES_PATH.exists() or not IHSG_PATH.exists() or not FEATURE_FRAME_PATH.exists():
        raise FileNotFoundError("Build articles.parquet and the Day-4 feature frame before EDA.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eligibility = pd.read_parquet(
        ARTICLES_PATH,
        columns=["article_id", "analysis_eligibility"],
    )
    ihsg = pd.read_parquet(IHSG_PATH)
    articles = pd.read_parquet(
        FEATURE_FRAME_PATH,
        columns=[
            "article_id",
            "source",
            "published_at",
            "content_token_count",
            "inset_sentiment_density",
        ],
    )
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    articles["published_at"] = pd.to_datetime(articles["published_at"])
    ihsg["date"] = pd.to_datetime(ihsg["date"]).dt.normalize()
    articles["date"] = articles["published_at"].dt.normalize()
    articles = articles.merge(eligibility, on="article_id", how="left", validate="one_to_one")
    articles = articles.rename(columns={"content_token_count": "content_tokens"})

    scorer = _load_inset()
    if articles["inset_sentiment_density"].isna().any():
        raise ValueError("EDA blocked: feature frame does not cover every article_id.")
    articles = articles.rename(columns={"inset_sentiment_density": "inset_sentiment"})
    source_order = articles["source"].value_counts().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    source_counts = articles["source"].value_counts().reindex(source_order)
    bars = axes[0].bar(source_order, source_counts.values, color=[SOURCE_COLORS.get(source, "#64748B") for source in source_order])
    axes[0].set_title("Komposisi Sumber Artikel")
    axes[0].set_ylabel("Jumlah artikel")
    for bar, count in zip(bars, source_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,.0f}", ha="center", va="bottom")
    upper = articles["content_tokens"].quantile(0.99)
    axes[1].hist(articles.loc[articles["content_tokens"] <= upper, "content_tokens"], bins=45, color="#7C3AED", edgecolor="white")
    axes[1].axvline(articles["content_tokens"].median(), color="#111827", linestyle="--", label="Median")
    axes[1].set_title("Panjang Isi (hingga persentil ke-99)")
    axes[1].set_xlabel("Jumlah token")
    axes[1].set_ylabel("Jumlah artikel")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "corpus_overview.png")

    monthly = articles.groupby([articles["published_at"].dt.to_period("M").astype(str), "source"], observed=True).size().unstack(fill_value=0)
    fig, axis = plt.subplots(figsize=(14, 5))
    monthly.plot.area(ax=axis, color=[SOURCE_COLORS.get(source, "#64748B") for source in monthly.columns], alpha=0.85)
    axis.set_title("Cakupan Artikel per Bulan dan Sumber")
    axis.set_xlabel("Bulan publikasi")
    axis.set_ylabel("Jumlah artikel")
    axis.legend(title="Sumber", frameon=False, ncol=len(monthly.columns), loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "temporal_coverage.png")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].hist(articles["inset_sentiment"], bins=45, color="#0F766E", edgecolor="white")
    axes[0].axvline(0, color="#111827", linewidth=1)
    axes[0].set_title("Distribusi Baseline Sentimen InSet")
    axes[0].set_xlabel("Sentiment density")
    axes[0].set_ylabel("Jumlah artikel")
    sns.boxplot(data=articles, x="source", y="inset_sentiment", order=source_order, hue="source", palette=SOURCE_COLORS, legend=False, showfliers=False, ax=axes[1])
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_title("Baseline Sentimen per Sumber")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Sentiment density")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("InSet adalah feature baseline, bukan label emosi", x=0.06, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "sentiment_baseline.png")

    daily_news = articles.groupby("date").agg(news_volume=("article_id", "count"), avg_sentiment=("inset_sentiment", "mean")).reset_index()
    merged = ihsg.merge(daily_news, on="date", how="inner").sort_values("date")
    merged["news_volume_7d"] = merged["news_volume"].rolling(7, min_periods=1).mean()
    merged["sentiment_7d"] = merged["avg_sentiment"].rolling(7, min_periods=1).mean()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis, column, label, color in (
        (axes[0], "close_price", "IHSG close", "#2563EB"),
        (axes[1], "news_volume_7d", "Volume berita (7 hari)", "#F97316"),
        (axes[2], "sentiment_7d", "Sentimen InSet (7 hari)", "#059669"),
    ):
        axis.plot(merged["date"], merged[column], color=color, linewidth=1.8)
        axis.axvspan(EVENT_START, EVENT_END, color="#F59E0B", alpha=0.12)
        axis.set_ylabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Tanggal")
    fig.suptitle("Konteks berita dan IHSG — deskriptif, bukan bukti kausal", x=0.08, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "market_context.png")

    source_mix, _, _ = _equal_complete_month_windows(articles)
    sentiment_return_correlation = merged["avg_sentiment"].corr(merged["daily_return"])
    volume_return_correlation = merged["news_volume"].corr(merged["daily_return"])
    metrics = {
        "dataset_version": DATASET_VERSION,
        "articles": {
            "row_count": len(articles),
            "required_field_nulls": dataset_manifest["quality"]["required_field_nulls"],
            "source_distribution": articles["source"].value_counts().sort_index().to_dict(),
            "published_at_min": articles["published_at"].min().isoformat(),
            "published_at_max": articles["published_at"].max().isoformat(),
            "content_token_median": float(articles["content_tokens"].median()),
            "content_token_p99": float(articles["content_tokens"].quantile(0.99)),
            "analysis_eligibility": articles["analysis_eligibility"].value_counts().to_dict(),
        },
        "inset": {
            "cross_polarity_terms_excluded": len(scorer.overlap_terms),
            "positive_phrases": scorer.positive_phrase_count,
            "negative_phrases": scorer.negative_phrase_count,
            "average_sentiment": float(articles["inset_sentiment"].mean()),
        },
        "temporal": {
            "source_mix_equal_complete_month_windows": source_mix,
            "august_2025_case_study_article_count": int(
                ((articles["date"] >= EVENT_START) & (articles["date"] <= EVENT_END)).sum()
            ),
        },
        "market": {
            "ihsg_rows": len(ihsg),
            "overlapping_dates": len(merged),
            "sentiment_return_correlation": 0.0 if pd.isna(sentiment_return_correlation) else float(sentiment_return_correlation),
            "volume_return_correlation": 0.0 if pd.isna(volume_return_correlation) else float(volume_return_correlation),
            "interpretation": "descriptive only; not evidence of causality",
        },
    }
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    logger.info("EDA complete at %s", OUTPUT_DIR)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_eda()
