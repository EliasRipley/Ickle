import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.train import _read_best_validation_loss, _save_bundle


class _TokenizerStub:
    def checkpoint_payload(self):
        return {"tokenizer": {"type": "test"}}


class TrainingCheckpointTests(unittest.TestCase):
    def test_atomic_bundle_persists_comparable_validation_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.best.pt"
            _save_bundle(
                str(path),
                model=torch.nn.Linear(2, 2),
                cfg=SimpleNamespace(n_embd=2),
                tokenizer=_TokenizerStub(),
                step=7,
                training_metrics={"validation_loss": 1.25, "best_step": 7},
            )
            self.assertEqual(_read_best_validation_loss(str(path)), 1.25)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_legacy_bundle_has_no_comparable_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"model_state": {}}, path)
            self.assertIsNone(_read_best_validation_loss(str(path)))


if __name__ == "__main__":
    unittest.main()
