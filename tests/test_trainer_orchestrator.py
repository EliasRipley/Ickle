import os
import tempfile
import unittest
from pathlib import Path

from src.trainer_orchestrator import TrainerOperator, TrainerProgram


class TrainerOrchestratorRollbackTests(unittest.TestCase):
    """Regression coverage for a real bug: on eval failure, rollback_on_fail
    only ever set a `"rollback": True` flag in the result dict -- it never
    actually reverted anything, since promotion only touches promote_to on
    success. The failed candidate file was left sitting at output_model,
    indistinguishable from a real candidate to any later run or reader.
    "auto-revert" now means quarantining the failed file out of the way."""

    def _make_operator(self, tmpdir, eval_result):
        model_path = os.path.join(tmpdir, "base.pt")
        output_model = os.path.join(tmpdir, "candidate.pt")
        Path(model_path).write_text("base")
        Path(output_model).write_text("candidate")

        program = TrainerProgram(
            program_id="run1",
            model_path=model_path,
            output_model=output_model,
            corpus_paths=[os.path.join(tmpdir, "corpus.txt")],
            promote_to=os.path.join(tmpdir, "promoted.pt"),
            rollback_on_fail=True,
        )
        Path(program.corpus_paths[0]).write_text("hello\n")

        op = TrainerOperator(
            runs_dir=os.path.join(tmpdir, "runs"),
            build_corpus_fn=lambda prog: {"corpus_path": prog.corpus_paths[0], "lines": 1},
            train_fn=lambda prog, emit: {"output_model": prog.output_model, "steps": prog.steps},
            eval_fn=lambda prog, emit: eval_result,
        )
        op.submit_program(program)
        return op, program

    def test_failed_eval_quarantines_candidate_instead_of_leaving_it_in_place(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op, program = self._make_operator(tmpdir, {"passed": False, "delta": -0.1})
            run = op.execute_run(program.program_id)

            self.assertEqual(run.status, "completed")
            result = run.steps["promote_or_rollback"]
            self.assertTrue(result["rollback"])
            self.assertFalse(result["promoted"])
            self.assertFalse(Path(program.output_model).exists())
            self.assertEqual(len(result["quarantined_files"]), 1)
            self.assertTrue(Path(result["quarantined_files"][0]).exists())
            self.assertFalse(Path(program.promote_to).exists())

    def test_passed_eval_promotes_and_leaves_nothing_quarantined(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            op, program = self._make_operator(tmpdir, {"passed": True, "delta": 0.2, "candidate_avg": 0.5})
            run = op.execute_run(program.program_id)

            self.assertEqual(run.status, "completed")
            result = run.steps["promote_or_rollback"]
            self.assertTrue(result["promoted"])
            self.assertTrue(Path(program.promote_to).exists())


if __name__ == "__main__":
    unittest.main()
