import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.features.build_feature_frame import _sentiment_density, _temporal_split, _term_count, _uppercase_ratio
from src.features.feature_contract import CLICKBAIT_TRIGGER_TERMS, MARKET_KEYWORDS, feature_contract
from src.features.lexicon_scoring import build_inset_scorer
from src.processing import label_governance
from src.processing.label_governance import _blank_template, _chaos_annotation_sample, _proportional_allocation


class TestDay3Day4Contracts(unittest.TestCase):
    def test_proportional_annotation_allocation_is_exact(self):
        sizes = pd.Series([10, 20, 30], index=["a", "b", "c"])
        allocation = _proportional_allocation(sizes, 12)
        self.assertEqual(int(allocation.sum()), 12)
        self.assertTrue((allocation > 0).all())

    def test_chaos_sample_is_bounded_exact_and_non_overlapping(self):
        dates = pd.date_range("2025-08-24", "2025-09-01", freq="D")
        articles = pd.DataFrame(
            [
                {
                    "article_id": f"{date:%Y%m%d}-{source}-{index}",
                    "source": source,
                    "published_at": date + pd.Timedelta(hours=index),
                    "title": "title",
                    "content_clean": "content",
                }
                for date in dates
                for source in ("detik", "kompas", "tempo")
                for index in range(3)
            ]
        )
        excluded = {"20250825-detik-0"}
        sample, manifest = _chaos_annotation_sample(articles, excluded, sample_size=21)
        published_at = pd.to_datetime(sample["published_at"])
        self.assertEqual(len(sample), 21)
        self.assertTrue(published_at.ge(pd.Timestamp("2025-08-25")).all())
        self.assertTrue(published_at.lt(pd.Timestamp("2025-09-01")).all())
        self.assertTrue(set(sample["article_id"]).isdisjoint(excluded))
        self.assertEqual(manifest["event_window_end_inclusive"], "2025-08-31")

    def test_finalizer_combines_both_non_overlapping_annotation_packages(self):
        outside_dates = pd.date_range("2025-01-01", periods=300, freq="D")
        event_dates = pd.date_range("2025-08-25", "2025-08-31", freq="D")
        rows = [
            {
                "article_id": f"outside-{index}",
                "source": ("detik", "kompas", "tempo")[index % 3],
                "published_at": date,
                "title": "title",
                "content_clean": "content",
            }
            for index, date in enumerate(outside_dates)
        ]
        rows.extend(
            {
                "article_id": f"event-{index}",
                "source": ("detik", "kompas", "tempo")[index % 3],
                "published_at": event_dates[index % len(event_dates)] + pd.Timedelta(minutes=index),
                "title": "title",
                "content_clean": "content",
            }
            for index in range(900)
        )
        articles = pd.DataFrame(rows)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            articles_path = root / "articles.parquet"
            output_path = root / "labels/finalized.parquet"
            manifest_path = root / "labels/finalization.json"
            articles.to_parquet(articles_path, index=False)
            expected = label_governance._expected_annotation_packages(articles)
            completed_paths = []
            for package_index, (package_name, (sample, _)) in enumerate(expected.items()):
                template = _blank_template(sample)
                for task in label_governance.LABEL_FIELDS:
                    template[f"{task}_label"] = ["yes" if index % 2 else "no" for index in range(len(template))]
                template["annotator_id"] = "annotator-1"
                template["annotation_status"] = "done" if package_index == 0 else "completed"
                completed_path = root / f"{package_name}.csv"
                template.to_csv(completed_path, index=False)
                completed_paths.append(completed_path)

            with (
                patch.object(label_governance, "PROJECT_ROOT", root),
                patch.object(label_governance, "GOVERNANCE_DIR", root / "labels"),
                patch.object(label_governance, "ARTICLES_PATH", articles_path),
                patch.object(label_governance, "FINALIZED_LABELS_PATH", output_path),
                patch.object(label_governance, "FINALIZATION_MANIFEST_PATH", manifest_path),
            ):
                manifest = label_governance.finalize_single_annotator_labels(completed_paths)

            finalized = pd.read_parquet(output_path)
            self.assertEqual(manifest["finalized_labels"]["row_count"], 600)
            self.assertEqual(finalized["article_id"].nunique(), 600)
            self.assertEqual(set(finalized["annotation_status"]), {"completed"})

    def test_temporal_split_never_mixes_future_into_train(self):
        dates = pd.Series(pd.date_range("2025-01-01", periods=20, freq="D"))
        split, plan = _temporal_split(dates)
        train_dates = dates[split == "train"]
        validation_dates = dates[split == "validation"]
        test_dates = dates[split == "test"]
        self.assertLessEqual(train_dates.max(), validation_dates.min())
        self.assertLessEqual(validation_dates.max(), test_dates.min())
        self.assertEqual(plan["method"], "chronological split by unique publication dates; no random sampling")

    def test_lexical_features_are_bounded_and_deterministic(self):
        self.assertEqual(_term_count("HEBOH, terungkap!", CLICKBAIT_TRIGGER_TERMS), 2)
        self.assertEqual(_term_count("Bank Indonesia menaikkan suku bunga.", MARKET_KEYWORDS), 2)
        self.assertEqual(_uppercase_ratio("ABC def"), 0.5)
        self.assertEqual(_sentiment_density("baik buruk", {"baik": 5}, {"buruk": -5}), 0.0)

    def test_inset_phrase_matching_and_cross_polarity_policy_are_explicit(self):
        scorer = build_inset_scorer(
            {"putus tali gantung": 5, "aib": 1},
            {"putus": -1, "aib": -5},
        )
        self.assertGreater(_sentiment_density("putus tali gantung", scorer), 0.0)
        self.assertEqual(_sentiment_density("aib", scorer), 0.0)
        self.assertEqual(len(scorer.overlap_terms), 1)

    def test_contract_excludes_future_market_targets_from_model_inputs(self):
        contract = feature_contract()
        definitions = {item["name"]: item for item in contract["definitions"]}
        self.assertFalse(definitions["ihsg_return_t1"]["allowed_in_day5_input"])
        self.assertIn("fit on train", contract["leakage_policy"][0].lower())


if __name__ == "__main__":
    unittest.main()
