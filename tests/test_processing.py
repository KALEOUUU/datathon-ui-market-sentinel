import os
import sys
import unittest

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.cleaning import clean_text_pipeline, clean_html, extract_article_structure, remove_boilerplate
from src.processing.build_dataset import canonicalize_url, parse_indonesian_date

class TestTextProcessing(unittest.TestCase):

    def test_clean_html(self):
        html_input = "<p>Halo <b>Dunia</b>! Ini website berita.</p>"
        self.assertEqual(" ".join(clean_html(html_input).split()), "Halo Dunia ! Ini website berita.")

    def test_remove_boilerplate(self):
        text = "IHSG melemah hari ini. Baca juga: IHSG Anjlok 5 Persen pada hari ini. Simak Video: Video Berita Saham."
        cleaned = remove_boilerplate(text)
        self.assertNotIn("Baca juga:", cleaned)
        self.assertNotIn("Simak Video:", cleaned)
        self.assertIn("IHSG melemah hari ini.", cleaned)

    def test_url_canonicalization_removes_tracking_only(self):
        url = "HTTPS://Example.com/news/?utm_source=x&ref=homepage#section"
        self.assertEqual(canonicalize_url(url), "https://example.com/news?ref=homepage")

    def test_boilerplate_does_not_truncate_following_article_text(self):
        text = "Pembuka artikel. Baca juga: Tautan terkait. Isi artikel setelah tautan tetap penting."
        cleaned = clean_text_pipeline(text)
        self.assertIn("Isi artikel setelah tautan tetap penting.", cleaned)

    def test_indonesian_date_parser_preserves_dataset_date(self):
        self.assertEqual(parse_indonesian_date("12/02/2025").date().isoformat(), "2025-02-12")
        self.assertEqual(parse_indonesian_date("4 Oktober 2025 | 14.00 WIB").isoformat(), "2025-10-04T14:00:00")
        self.assertIsNone(parse_indonesian_date("tanggal tidak valid"))

    def test_source_aware_preprocessing_separates_dateline_and_quotes(self):
        article = 'AKARTA, KOMPAS.com - Hevearita Gunaryanti Rahayu (HGR) alias MbakIta berkata, "Situasi aman."'
        result = extract_article_structure(article, "kompas")
        self.assertEqual(result["dateline_location"], "JAKARTA")
        self.assertEqual(result["dateline_publisher"], "kompas.com")
        self.assertNotIn("KOMPAS.com", result["content_clean"])
        self.assertEqual(result["quote_count"], 1)
        self.assertEqual(result["quoted_text"], "Situasi aman.")
        self.assertIn("Hevearita Gunaryanti Rahayu", result["content_entity_normalized"])
        self.assertNotIn("MbakIta", result["content_entity_normalized"])

    def test_tempo_navigation_text_is_not_semantic_content(self):
        article = "Baca berita dengan sedikit iklan, klik di sini INFO NASIONAL - Isi utama. Scroll ke bawah untuk melanjutkan membaca"
        result = extract_article_structure(article, "tempo")
        self.assertEqual(result["section_label"], "INFO NASIONAL")
        self.assertEqual(result["content_clean"], "Isi utama.")

    def test_unverified_acronym_is_not_expanded_by_heuristic(self):
        result = extract_article_structure("Komisi Pemberantasan Korupsi (KPK) memeriksa saksi.", "kompas")
        self.assertEqual(result["entity_aliases_json"], "{}")
        self.assertIn("(KPK)", result["content_entity_normalized"])

    def test_consecutive_duplicate_datelines_are_all_removed(self):
        result = extract_article_structure("JAKARTA, KOMPAS.com - JAKARTA, KOMPAS.com – Isi artikel yang valid.", "kompas")
        self.assertEqual(result["content_clean"], "Isi artikel yang valid.")
        self.assertEqual(result["dateline_location"], "JAKARTA")

    def test_source_specific_dateline_variants_are_metadata_not_content(self):
        kompas = extract_article_structure("JAKARTA, KOMPAS,com - Isi artikel.", "kompas")
        tempo = extract_article_structure("JAKARTA, Tempo.co--Isi artikel.", "tempo")
        tempo_reverse = extract_article_structure("TEMPO.CO, Jakarta - Isi artikel.", "tempo")
        self.assertEqual(kompas["content_clean"], "Isi artikel.")
        self.assertEqual(kompas["dateline_publisher"], "kompas.com")
        self.assertEqual(tempo["content_clean"], "Isi artikel.")
        self.assertEqual(tempo["dateline_publisher"], "tempo.co")
        self.assertEqual(tempo_reverse["content_clean"], "Isi artikel.")
        self.assertEqual(tempo_reverse["dateline_location"], "JAKARTA")

    def test_ambiguous_tempo_hyphen_prefix_is_preserved_as_article_text(self):
        result = extract_article_structure("TIMNAS U-17 Indonesia menang.", "tempo")
        self.assertIsNone(result["section_label"])
        self.assertEqual(result["content_clean"], "TIMNAS U-17 Indonesia menang.")

if __name__ == "__main__":
    unittest.main()
