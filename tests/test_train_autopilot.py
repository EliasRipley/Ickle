import unittest
from pathlib import Path
from uuid import uuid4

import torch

from src.train_autopilot import _checkpoint_step


class TrainAutopilotTests(unittest.TestCase):
    def test_checkpoint_step_missing_returns_minus_one(self):
        p = Path("data/.tmp_tests") / f"missing_{uuid4().hex}.pt"
        self.assertEqual(_checkpoint_step(p), -1)

    def test_checkpoint_step_reads_step_field(self):
        root = Path("data/.tmp_tests")
        root.mkdir(parents=True, exist_ok=True)
        p = root / f"ckpt_{uuid4().hex}.pt"
        torch.save({"step": 321, "model_state": {}, "config": {}}, str(p))
        self.assertEqual(_checkpoint_step(p), 321)


if __name__ == "__main__":
    unittest.main()
