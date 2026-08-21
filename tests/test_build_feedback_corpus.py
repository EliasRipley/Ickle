import json
import tempfile
import unittest
from pathlib import Path

from src.build_feedback_corpus import build_corpus


class BuildFeedbackCorpusTests(unittest.TestCase):
    """Regression coverage for a real bug: build_corpus called json.loads(line)
    with no error handling, unlike its sibling build_preference_pairs.py --
    a single malformed line anywhere in the feedback JSONL file crashed the
    entire corpus build instead of just skipping that row."""

    def test_malformed_line_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_path = Path(tmpdir) / "feedback.jsonl"
            out_path = Path(tmpdir) / "corpus.txt"
            lines = [
                json.dumps({"prompt": "hi", "response": "hello", "rating": 5}),
                "{not valid json!!",
                json.dumps({"prompt": "bye", "response": "goodbye", "rating": 5}),
            ]
            feedback_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            build_corpus(str(feedback_path), str(out_path), min_rating=4)

            text = out_path.read_text(encoding="utf-8")
            self.assertIn("User: hi", text)
            self.assertIn("User: bye", text)

    def test_malformed_rating_falls_back_to_zero_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_path = Path(tmpdir) / "feedback.jsonl"
            out_path = Path(tmpdir) / "corpus.txt"
            lines = [
                json.dumps({"prompt": "hi", "response": "hello", "rating": "not-a-number"}),
                json.dumps({"prompt": "bye", "response": "goodbye", "rating": 5}),
            ]
            feedback_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            build_corpus(str(feedback_path), str(out_path), min_rating=4)

            text = out_path.read_text(encoding="utf-8")
            self.assertNotIn("User: hi", text)
            self.assertIn("User: bye", text)


if __name__ == "__main__":
    unittest.main()
