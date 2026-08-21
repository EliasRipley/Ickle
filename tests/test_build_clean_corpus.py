import json
import unittest
import uuid
from pathlib import Path

from src.build_clean_corpus import build_corpus


class BuildCleanCorpusTests(unittest.TestCase):
    @staticmethod
    def _tmp_training_root() -> Path:
        root = Path("data") / ".tmp_tests" / f"corpus_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_build_corpus_includes_dictionary_dialogs(self):
        root = self._tmp_training_root()
        dictionary_path = root / "webster_dictionary.json"
        dictionary_path.write_text(
            json.dumps(
                {
                    "event horizon": "The boundary beyond which events cannot affect an outside observer.",
                    "tectonics": "The study of the structure and movement of Earth's crust.",
                }
            ),
            encoding="utf-8",
        )
        lines, stats = build_corpus(training_root=root, max_lines=200, dictionary_items=10)
        joined = "\n".join(lines)
        self.assertIn("What does event horizon mean?", joined)
        self.assertIn("What does tectonics mean?", joined)
        self.assertGreater(stats["dictionary_lines"], 0)


if __name__ == "__main__":
    unittest.main()

