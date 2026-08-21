import os
import tempfile
import unittest
from pathlib import Path

from src.train_invoke import build_train_command


class BuildTrainCommandTests(unittest.TestCase):
    """Regression coverage for a real bug: continual_guard.py, train_autopilot.py,
    train_cycle.py, train_intelligence_stack.py, and trainer_orchestrator.py each
    hand-built the `src.train` argv independently. Only continual_guard.py and
    train_autopilot.py actually supported --resume-from-checkpoint; the other
    three (including the base stage of train_intelligence_stack.py, which does
    have a checkpoint path) always restarted from scratch/--init-model even when
    a resumable checkpoint already existed on disk."""

    def test_resumes_when_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.pt.checkpoint.pt")
            Path(checkpoint_path).write_text("fake checkpoint")

            cmd = build_train_command(
                data_path="corpus.txt",
                out_model="out.pt",
                steps=100,
                init_model="base.pt",
                checkpoint_path=checkpoint_path,
                resume_if_possible=True,
            )
            self.assertIn("--resume-from-checkpoint", cmd)
            self.assertEqual(cmd[cmd.index("--resume-from-checkpoint") + 1], checkpoint_path)
            self.assertNotIn("--init-model", cmd)

    def test_falls_back_to_init_model_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "does_not_exist.pt.checkpoint.pt")
            cmd = build_train_command(
                data_path="corpus.txt",
                out_model="out.pt",
                steps=100,
                init_model="base.pt",
                checkpoint_path=checkpoint_path,
                resume_if_possible=True,
            )
            self.assertNotIn("--resume-from-checkpoint", cmd)
            self.assertIn("--init-model", cmd)
            self.assertEqual(cmd[cmd.index("--init-model") + 1], "base.pt")

    def test_falls_back_to_tokenizer_setup_when_no_init_model_or_checkpoint(self):
        cmd = build_train_command(
            data_path="corpus.txt",
            out_model="out.pt",
            steps=100,
            tokenizer="sentencepiece",
            spm_vocab_size=2048,
            spm_model_type="bpe",
        )
        self.assertNotIn("--resume-from-checkpoint", cmd)
        self.assertNotIn("--init-model", cmd)
        self.assertIn("--tokenizer", cmd)
        self.assertEqual(cmd[cmd.index("--tokenizer") + 1], "sentencepiece")

    def test_resume_if_possible_false_ignores_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.pt.checkpoint.pt")
            Path(checkpoint_path).write_text("fake checkpoint")
            cmd = build_train_command(
                data_path="corpus.txt",
                out_model="out.pt",
                steps=100,
                init_model="base.pt",
                checkpoint_path=checkpoint_path,
                resume_if_possible=False,
            )
            self.assertNotIn("--resume-from-checkpoint", cmd)
            self.assertIn("--init-model", cmd)

if __name__ == "__main__":
    unittest.main()
