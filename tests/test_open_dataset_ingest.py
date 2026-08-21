import unittest

from src.open_dataset_ingest import _clean_text, _extract_text_field


class OpenDatasetIngestTests(unittest.TestCase):
    def test_clean_text_normalizes_whitespace_and_terminal_punctuation(self):
        out = _clean_text("  hello   world  ", max_chars=100)
        self.assertEqual(out, "hello world.")

    def test_extract_uses_preferred_field(self):
        row = {"text": "A" * 120, "other": "short"}
        out = _extract_text_field(row, preferred_field="text", max_chars=80)
        self.assertTrue(out.startswith("A" * 80))

    def test_extract_falls_back_to_first_long_string(self):
        row = {"meta": 1, "body": "B" * 100}
        out = _extract_text_field(row, preferred_field="text", max_chars=90)
        self.assertTrue(out.startswith("B" * 90))


if __name__ == "__main__":
    unittest.main()
