"""Versioned Day-4 feature contract; definitions only, no fitted ML transformer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_VERSION = "v1"
CONTRACT_PATH = PROJECT_ROOT / "data/processed/features/feature_contract.json"

# These are the only examples explicitly named in docs/project_context.md.
# They are weak lexical indicators, not labels and not a complete lexicon.
CLICKBAIT_TRIGGER_TERMS = ("terungkap", "heboh", "waspada")
MARKET_KEYWORDS = ("ihsg", "rupiah", "suku bunga", "dolar", "bi", "bank indonesia", "jci", "wall street", "dow jones", "ftse", "bursa efek", "lq45")


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    source: str
    availability: str
    description: str
    allowed_in_day5_input: bool = True


FEATURE_DEFINITIONS = (
    FeatureDefinition("title_char_count", "title", "at_publication", "Jumlah karakter judul."),
    FeatureDefinition("title_token_count", "title", "at_publication", "Jumlah token judul dengan tokenisasi whitespace konsisten."),
    FeatureDefinition("title_exclamation_count", "title", "at_publication", "Jumlah tanda seru pada judul."),
    FeatureDefinition("title_question_count", "title", "at_publication", "Jumlah tanda tanya pada judul."),
    FeatureDefinition("title_uppercase_ratio", "title", "at_publication", "Rasio huruf kapital terhadap seluruh huruf pada judul."),
    FeatureDefinition("title_digit_count", "title", "at_publication", "Jumlah digit pada judul."),
    FeatureDefinition("title_clickbait_trigger_count", "title", "at_publication", "Jumlah kemunculan tiga contoh kata pemicu yang didokumentasikan; bukan label."),
    FeatureDefinition("content_token_count", "content_clean", "at_publication", "Jumlah token isi artikel."),
    FeatureDefinition("content_sentence_proxy_count", "content_clean", "at_publication", "Jumlah pemisah kalimat sebagai proxy struktur teks."),
    FeatureDefinition("content_quote_count", "quote_count", "at_publication", "Jumlah segmen kutipan langsung yang diekstrak secara deterministik saat preprocessing."),
    FeatureDefinition("content_digit_ratio", "content_clean", "at_publication", "Rasio digit terhadap panjang isi artikel."),
    FeatureDefinition("inset_sentiment_density", "content_clean + InSet", "at_publication", "Baseline sentimen lexicon terikat [-1, 1]; bukan ground-truth emosi."),
    FeatureDefinition("inset_sentiment_extremity", "inset_sentiment_density", "at_publication", "Nilai absolut baseline sentimen; kandidat feature, bukan final risk component."),
    FeatureDefinition("market_keyword_count", "title + content_clean", "at_publication", "Kemunculan kata kunci pasar yang didokumentasikan; bukan market_sensitivity label."),
    FeatureDefinition("market_keyword_present", "market_keyword_count", "at_publication", "Indikator setidaknya satu kata kunci pasar."),
    FeatureDefinition("tfidf_title_content", "title + content_clean", "fit_on_train_only", "Sparse n-gram matrix; belum dimaterialisasi di Hari 4 untuk mencegah leakage."),
    FeatureDefinition("ihsg_return_t1", "ihsg_data", "after_publication", "Target/analisis market masa depan; dilarang menjadi input model artikel." , False),
    FeatureDefinition("realized_volatility_future", "ihsg_data", "after_publication", "Target analisis market masa depan; dilarang menjadi input model artikel." , False),
)


def feature_contract() -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "entity_key": "article_id",
        "time_key": "published_at",
        "definitions": [asdict(item) for item in FEATURE_DEFINITIONS],
        "lexical_indicator_policy": {
            "clickbait_trigger_terms": list(CLICKBAIT_TRIGGER_TERMS),
            "market_keywords": list(MARKET_KEYWORDS),
            "limitation": "Terms are documented examples only. They support exploratory features and must not become pseudo-labels.",
        },
        "inset_scoring_policy": {
            "matching": "case-normalized longest phrase match without overlap",
            "cross_polarity_terms": "excluded from both polarities because InSet alone cannot disambiguate their senses",
            "limitation": "This remains a lexicon baseline and is not a ground-truth emotion label.",
        },
        "leakage_policy": [
            "Any transformer with learned vocabulary/statistics, including TF-IDF, is fit on train split only in Day 5.",
            "Future IHSG values are targets/analysis fields only and are excluded from article-level feature inputs.",
            "No label-derived aggregate is available to a feature row at publication time.",
        ],
        "deferred_until_validated": [
            "Named-entity features require a documented Indonesian NER component.",
            "Semantic embeddings require a chosen and versioned encoder.",
            "Label-dependent score calibration requires finalized human labels with their governance status recorded.",
        ],
    }


def write_feature_contract() -> dict:
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    contract = feature_contract()
    with CONTRACT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
    return contract


if __name__ == "__main__":
    write_feature_contract()
