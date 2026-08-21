import unittest

from src.build_clean_corpus import _is_low_quality_pair
from src.data_quality import CONTENT_STOPWORDS, ENGLISH_HINT_TOKENS, dialogue_pair_fails_content_checks
from src.open_dataset_ingest import _pair_is_low_quality


class SharedLowQualityPairChecksTests(unittest.TestCase):
    """Regression coverage for real duplication: build_clean_corpus.py and
    open_dataset_ingest.py each carried a byte-for-byte-identical copy of
    ENGLISH_HINT_TOKENS/CONTENT_STOPWORDS and the ascii-ratio/topic-overlap/
    repeated-word-run/diversity tail of their low-quality-pair heuristic.
    A fix to one never reached the other. Both now delegate to
    data_quality.dialogue_pair_fails_content_checks; this asserts they still
    agree on borderline cases."""

    def test_constants_are_the_single_shared_definition(self):
        self.assertIn("the", ENGLISH_HINT_TOKENS)
        self.assertIn("what", CONTENT_STOPWORDS)

    def test_both_builders_reject_non_english_pair(self):
        prompt = "Bonjour, comment vas-tu aujourd'hui mon ami"
        response = "Je vais bien merci beaucoup pour ta question amicale"
        self.assertTrue(_is_low_quality_pair(prompt, response))
        self.assertTrue(_pair_is_low_quality(prompt, response))

    def test_both_builders_reject_unrelated_pair(self):
        prompt = "What is the capital of France for geography homework"
        response = "Bananas taste great in smoothies with yogurt and honey added"
        self.assertTrue(_is_low_quality_pair(prompt, response))
        self.assertTrue(_pair_is_low_quality(prompt, response))

    def test_both_builders_accept_good_pair(self):
        prompt = "Can you explain how photosynthesis works in plants"
        response = "Plants convert sunlight, water, and carbon dioxide into glucose and oxygen through photosynthesis."
        self.assertFalse(_is_low_quality_pair(prompt, response))
        self.assertFalse(_pair_is_low_quality(prompt, response))

    def test_shared_helper_directly(self):
        self.assertTrue(dialogue_pair_fails_content_checks(
            "hello there friend", "wat wat wat wat wat wat wat wat wat wat wat wat"
        ))


if __name__ == "__main__":
    unittest.main()
