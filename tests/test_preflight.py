import unittest
import tempfile
from pathlib import Path
from unittest import mock

from src.preflight_win11 import _check_gpu, _check_promotion_benchmark, _check_training_root


class PreflightTests(unittest.TestCase):
    def test_cpu_only_is_ready_but_explains_tradeoff(self):
        with mock.patch("src.device_bridge.get_gpu_info", return_value={"available": False}):
            ok, message = _check_gpu()
        self.assertTrue(ok)
        self.assertIn("CPU-only", message)

    def test_default_promotion_benchmark_is_ready(self):
        ok, message = _check_promotion_benchmark()
        self.assertTrue(ok, message)

    def test_training_root_rejects_tiny_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tiny.txt").write_text("too small", encoding="utf-8")
            with mock.patch("src.workspace_paths.get_training_root", return_value=root):
                ok, message = _check_training_root()
        self.assertFalse(ok)
        self.assertIn("need at least", message)


if __name__ == "__main__":
    unittest.main()
