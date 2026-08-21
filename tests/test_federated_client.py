import os
import tempfile
import unittest
from pathlib import Path

from src.federated.client import _pick_training_corpus, _resolve_eval_data_path, _resolve_local_data_path


class FederatedClientPathTests(unittest.TestCase):
    def test_pick_training_corpus_prefers_known_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp)
            (training_root / "open_fineweb_edu_stream.txt").write_text("open", encoding="utf-8")
            (training_root / "combined_corpus.txt").write_text("combined", encoding="utf-8")
            picked = _pick_training_corpus(training_root)
            self.assertEqual(picked, training_root / "combined_corpus.txt")

    def test_resolve_local_data_explicit_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp) / "IckleTraining"
            training_root.mkdir(parents=True, exist_ok=True)
            explicit = Path(tmp) / "my_data.txt"
            explicit.write_text("hello", encoding="utf-8")

            resolved, origin = _resolve_local_data_path(
                str(explicit),
                training_root=training_root,
                prefer_training_root_data=True,
            )
            self.assertEqual(resolved, explicit.resolve())
            self.assertEqual(origin, "explicit")

    def test_resolve_local_data_prefers_training_root_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp) / "IckleTraining"
            training_root.mkdir(parents=True, exist_ok=True)
            expected = training_root / "combined_corpus.txt"
            expected.write_text("train", encoding="utf-8")

            resolved, origin = _resolve_local_data_path(
                "",
                training_root=training_root,
                prefer_training_root_data=True,
            )
            self.assertEqual(resolved, expected.resolve())
            self.assertEqual(origin, "training_root")

    def test_resolve_local_data_uses_legacy_default_for_long_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                legacy = Path("data") / "ickle_clean_corpus.txt"
                legacy.parent.mkdir(parents=True, exist_ok=True)
                legacy.write_text("legacy", encoding="utf-8")

                training_root = Path(tmp) / "IckleTraining"
                training_root.mkdir(parents=True, exist_ok=True)
                (training_root / "combined_corpus.txt").write_text("train", encoding="utf-8")

                resolved, origin = _resolve_local_data_path(
                    "",
                    training_root=training_root,
                    prefer_training_root_data=False,
                )
                expected_path = legacy.resolve()
            finally:
                os.chdir(original_cwd)

            self.assertEqual(resolved, expected_path)
            self.assertEqual(origin, "legacy_default")

    def test_resolve_local_data_falls_back_to_training_root_when_legacy_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                training_root = Path(tmp) / "IckleTraining"
                training_root.mkdir(parents=True, exist_ok=True)
                expected = training_root / "ickle_v3_corpus.txt"
                expected.write_text("train", encoding="utf-8")

                resolved, origin = _resolve_local_data_path(
                    "",
                    training_root=training_root,
                    prefer_training_root_data=False,
                )
            finally:
                os.chdir(original_cwd)

            self.assertEqual(resolved, expected.resolve())
            self.assertEqual(origin, "training_root_fallback")

    def test_resolve_eval_data_defaults_to_local_data(self):
        local = Path("example_eval.txt").resolve()
        resolved = _resolve_eval_data_path("", local_data_path=local)
        self.assertEqual(resolved, local)


if __name__ == "__main__":
    unittest.main()
