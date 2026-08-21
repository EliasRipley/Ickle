import os
import subprocess
import sys
import tempfile
import unittest

import torch

from src.model import ILM, TinyConfig

try:
    import onnx
    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False


@unittest.skipUnless(HAVE_ONNX, "onnx package not installed (see requirements-onnx.txt)")
class ExportOnnxTests(unittest.TestCase):
    """Regression coverage for a real bug: export_onnx.py declared
    output_names=["logits", "loss"], but ILM.forward(idx) with no `targets`
    (the only call shape that makes sense for an inference export) returns
    (logits, None) -- a None output torch.onnx.export cannot trace. The
    command also had no requirements entry for the `onnx` package it needs
    to write the file, so it had never actually run successfully."""

    def test_cli_exports_loadable_onnx_with_logits_output(self):
        cfg = TinyConfig(vocab_size=64, block_size=32, n_embd=16, n_head=2, n_layer=2)
        model = ILM(cfg)

        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "tiny.pt")
            out_path = os.path.join(d, "out.onnx")
            torch.save({"config": cfg.__dict__, "model_state": model.state_dict()}, ckpt_path)

            result = subprocess.run(
                [sys.executable, "-m", "src.export_onnx", "--model", ckpt_path, "--out", out_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.isfile(out_path))

            onnx_model = onnx.load(out_path)
            onnx.checker.check_model(onnx_model)
            self.assertEqual([o.name for o in onnx_model.graph.output], ["logits"])
            self.assertEqual([i.name for i in onnx_model.graph.input], ["input_ids"])


if __name__ == "__main__":
    unittest.main()
