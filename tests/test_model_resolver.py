import os
import tempfile
import time
import unittest
from pathlib import Path

from src.model_resolver import resolve_default_model
from src.runtime_flags import RuntimeFlagsStore


class ModelResolverTests(unittest.TestCase):
    def test_prefers_runtime_flag_current_model(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            os.chdir(td)
            try:
                models = Path("models")
                models.mkdir(parents=True, exist_ok=True)
                a = models / "a.pt"
                b = models / "b.pt"
                a.write_text("a", encoding="utf-8")
                b.write_text("b", encoding="utf-8")
                now = time.time()
                os.utime(a, (now - 60, now - 60))
                os.utime(b, (now, now))

                flags = RuntimeFlagsStore(path="data/runtime_flags.json")
                flags.update_flags({"current_model": str(a)})

                self.assertEqual(resolve_default_model(), str(a.resolve().as_posix()))
            finally:
                os.chdir(cwd)

    def test_falls_back_to_latest_model_when_flag_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            os.chdir(td)
            try:
                models = Path("models")
                models.mkdir(parents=True, exist_ok=True)
                a = models / "older.pt"
                b = models / "newer.pt"
                a.write_text("a", encoding="utf-8")
                b.write_text("b", encoding="utf-8")
                now = time.time()
                os.utime(a, (now - 60, now - 60))
                os.utime(b, (now, now))

                flags = RuntimeFlagsStore(path="data/runtime_flags.json")
                flags.update_flags({"current_model": "models/missing.pt"})

                self.assertEqual(resolve_default_model(), str(b.resolve().as_posix()))
            finally:
                os.chdir(cwd)

    def test_finds_models_in_candidates_subdirectory(self):
        """Regression test for a real bug: every training task (including
        the web UI's own "Start training" flow) writes its output to
        models/candidates/, but resolve_default_model() used to only glob
        models/ directly -- a freshly, successfully trained model was
        invisible to chat entirely unless manually moved. Confirmed live:
        a completed training run's model sat unusable until this was fixed."""
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            os.chdir(td)
            try:
                candidates = Path("models/candidates")
                candidates.mkdir(parents=True, exist_ok=True)
                trained = candidates / "ickle_baseline.pt"
                trained.write_text("trained", encoding="utf-8")

                self.assertEqual(resolve_default_model(), str(trained.resolve().as_posix()))
            finally:
                os.chdir(cwd)

    def test_prefers_newer_model_regardless_of_which_directory(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            os.chdir(td)
            try:
                models = Path("models")
                candidates = models / "candidates"
                candidates.mkdir(parents=True, exist_ok=True)
                older_promoted = models / "older.pt"
                newer_candidate = candidates / "newer.pt"
                older_promoted.write_text("a", encoding="utf-8")
                newer_candidate.write_text("b", encoding="utf-8")
                now = time.time()
                os.utime(older_promoted, (now - 60, now - 60))
                os.utime(newer_candidate, (now, now))

                self.assertEqual(resolve_default_model(), str(newer_candidate.resolve().as_posix()))
            finally:
                os.chdir(cwd)

    def test_ignores_checkpoint_files_in_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            os.chdir(td)
            try:
                candidates = Path("models/candidates")
                candidates.mkdir(parents=True, exist_ok=True)
                real_model = candidates / "ickle_baseline.pt"
                checkpoint = candidates / "ickle_baseline.pt.checkpoint.pt"
                real_model.write_text("model", encoding="utf-8")
                checkpoint.write_text("checkpoint", encoding="utf-8")
                now = time.time()
                os.utime(real_model, (now - 60, now - 60))
                os.utime(checkpoint, (now, now))  # newer, but must still be skipped

                self.assertEqual(resolve_default_model(), str(real_model.resolve().as_posix()))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
