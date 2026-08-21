import unittest

from src.build_base_lm_corpus import _clean_paragraph


class BuildBaseLmCorpusTests(unittest.TestCase):
    def test_clean_paragraph_rejects_code_like_text(self):
        text = "def hello(): print('x') " * 20
        self.assertEqual(_clean_paragraph(text, max_chars=800), "")

    def test_clean_paragraph_accepts_plain_text(self):
        text = (
            "Gorillas are ground-dwelling, predominantly herbivorous apes that inhabit the forests "
            "of central Sub-Saharan Africa. They are the largest living primates and live in family groups."
        )
        out = _clean_paragraph(text, max_chars=800)
        self.assertIn("Gorillas", out)
        self.assertGreater(len(out), 80)


if __name__ == "__main__":
    unittest.main()
