import json
import unittest
import uuid
from pathlib import Path

from src.build_preference_pairs import build_preference_pairs


class BuildPreferencePairsTests(unittest.TestCase):
    @staticmethod
    def _tmp_dir() -> Path:
        root = Path("data") / ".tmp_tests" / f"prefs_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_builds_pairs_from_rated_variants(self):
        root = self._tmp_dir()
        feedback = root / "feedback.jsonl"
        out = root / "pairs.jsonl"
        rows = [
            {"prompt": "Explain gravity", "response": "good answer", "rating": 5},
            {"prompt": "Explain gravity", "response": "bad answer", "rating": 1},
            {"prompt": "Say hi", "response": "hello", "rating": 3},
            {"prompt": "Say hi", "response": "hi there", "rating": 4},
        ]
        with feedback.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        stats = build_preference_pairs(
            str(feedback),
            str(out),
            min_rating_gap=2,
            min_chosen_rating=4,
        )

        self.assertEqual(stats["pairs_written"], 1)
        payloads = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(payloads), 1)
        pair = payloads[0]
        self.assertEqual(pair["prompt"], "Explain gravity")
        self.assertEqual(pair["chosen"], "good answer")
        self.assertEqual(pair["rejected"], "bad answer")


if __name__ == "__main__":
    unittest.main()

