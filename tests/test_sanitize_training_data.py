import unittest
from pathlib import Path
from uuid import uuid4

from src.sanitize_training_data import _parse_pairs, run_sanitize_training_data


class SanitizeTrainingDataTests(unittest.TestCase):
    def test_near_duplicate_pairs_are_removed_by_minhash(self):
        # Regression: dedup used to be exact-match-only (a `seen` set keyed
        # on lowercased text), so a rephrased near-duplicate pair sailed
        # straight through. minhash_deduplicate (already used elsewhere in
        # data_quality.py) catches high-similarity pairs like this one.
        user = "Can you explain how photosynthesis works in green plants during the day for my biology homework assignment"
        assistant = (
            "Plants convert sunlight, water, and carbon dioxide into glucose and oxygen through photosynthesis "
            "in their leaves using chlorophyll to capture light energy from the sun each day"
        )
        # A single trailing word changed -- the overwhelming majority of
        # 3-gram shingles in both the prompt and the (much longer) response
        # are identical, so the estimated Jaccard clears the 0.85 threshold.
        lines = [
            f"User: {user}",
            f"Ickle: {assistant}",
            "",
            f"User: {user}",
            f"Ickle: {assistant} today",
            "",
        ]
        pairs = _parse_pairs(lines, drop_qa_templates=False)
        self.assertEqual(len(pairs), 1)

    def test_sanitize_removes_duplicates_and_contamination(self):
        root = Path("data/.tmp_tests") / f"sanitize_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        target = root / "memory_aware_training.txt"
        target.write_text(
            "\n".join(
                [
                    "User: Who created you?",
                    "Ickle: My creator is Elias Ripley.",
                    "",
                    "User: What is entropy?",
                    "Ickle: Entropy measures disorder in a system over time.",
                    "",
                    "User: What is entropy?",
                    "Ickle: Entropy measures disorder in a system over time.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = run_sanitize_training_data(
            training_root=root,
            apply=True,
            drop_qa_templates=False,
            include_files=["memory_aware_training.txt"],
        )
        self.assertEqual(report["changed_files"], 1)
        text = target.read_text(encoding="utf-8").lower()
        self.assertNotIn("elias ripley", text)
        self.assertEqual(text.count("what is entropy?"), 1)


if __name__ == "__main__":
    unittest.main()

