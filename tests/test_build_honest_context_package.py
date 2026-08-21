import json
import unittest
from pathlib import Path
from uuid import uuid4

from src.build_honest_context_package import build_honest_context_package


class BuildHonestContextPackageTests(unittest.TestCase):
    @staticmethod
    def _tmpdir() -> Path:
        root = Path("data/.tmp_tests") / f"honest_pkg_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=False)
        return root

    def test_build_package_filters_hardcoded_identity_and_low_signal_rows(self):
        td = self._tmpdir()
        training_root = td / "training"
        training_root.mkdir(parents=True, exist_ok=True)

        (training_root / "open_openhermes_2_5_stream.txt").write_text(
            "\n".join(
                [
                    "User: If you are not sure, what should you do?",
                    "Ickle: I should say I am uncertain and verify with a reliable source.",
                    "",
                    "User: Who created you?",
                    "Ickle: My creator is Elias Ripley.",
                    "",
                    "User: Can you help me?",
                    "Ickle: I can help with that. Share the exact outcome you want and any constraints.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        out_dir = td / "out"
        report = build_honest_context_package(
            training_root=training_root,
            out_dir=out_dir,
            max_source_pairs=120,
            target_sft_pairs=150,
            seed=7,
        )

        self.assertGreaterEqual(int(report["selected_sft_pairs"]), 80)
        self.assertGreaterEqual(int(report["preference_rows"]), 70)

        sft_text = (out_dir / "honest_context_sft.txt").read_text(encoding="utf-8").lower()
        self.assertIn("uncertain", sft_text)
        self.assertNotIn("my creator is", sft_text)
        self.assertNotIn("elias ripley", sft_text)
        self.assertNotIn("share the exact outcome", sft_text)

        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("category_distribution", manifest)
        self.assertTrue((out_dir / "honest_context_eval_cases.json").exists())
        self.assertTrue((out_dir / "honest_context_preferences.jsonl").exists())

    def test_preference_rows_have_distinct_chosen_rejected(self):
        td = self._tmpdir()
        training_root = td / "training"
        training_root.mkdir(parents=True, exist_ok=True)
        (training_root / "open_openhermes_2_5_stream.txt").write_text(
            "\n".join(
                [
                    "User: The request is vague. What should you do first?",
                    "Ickle: I should ask one focused clarification question before acting.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        out_dir = td / "out"
        build_honest_context_package(
            training_root=training_root,
            out_dir=out_dir,
            max_source_pairs=80,
            target_sft_pairs=120,
            seed=11,
        )

        prefs_path = out_dir / "honest_context_preferences.jsonl"
        rows = [json.loads(line) for line in prefs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertNotEqual(
                str(row.get("chosen", "")).strip().lower(),
                str(row.get("rejected", "")).strip().lower(),
            )


if __name__ == "__main__":
    unittest.main()
