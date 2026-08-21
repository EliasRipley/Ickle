import json
import tempfile
import unittest
from pathlib import Path

from src.autodidact import build_python_autodidact_corpus


class AutodidactTests(unittest.TestCase):
    def test_keeps_only_objectively_good_rows(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "attempts.jsonl"
            out = Path(td) / "corpus.txt"
            rows = [
                {"prompt": "solve a", "response": "code a", "tests_passed": True, "lint_passed": True},
                {"prompt": "solve b", "response": "code b", "tests_passed": False, "lint_passed": True},
                {"prompt": "solve c", "response": "code c", "tests_passed": True, "lint_passed": False},
            ]
            with log.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            kept = build_python_autodidact_corpus(str(log), str(out))
            self.assertEqual(kept, 1)
            text = out.read_text(encoding="utf-8")
            self.assertIn("solve a", text)
            self.assertNotIn("solve b", text)


if __name__ == "__main__":
    unittest.main()
