import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.task_actions import (
    run_build_dpo_prefs_task,
    run_dpo_train_task,
    run_generate_teacher_data_task,
)


def _noop_progress(_msg: str) -> None:
    pass


class GenerateTeacherDataTaskTests(unittest.TestCase):
    """run_generate_teacher_data_task wires the web UI's Teach tab to the
    same TeacherBase.batch_sft()/submit_to_store() calls the anthropic-teach
    and registry-teach CLIs already use -- these tests cover the new
    validation/dispatch logic, not TeacherBase itself (already exercised by
    the CLI tools)."""

    def test_missing_topic_raises(self):
        with self.assertRaises(ValueError):
            run_generate_teacher_data_task({"provider": "anthropic"}, _noop_progress)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            run_generate_teacher_data_task({"topic": "x", "provider": "not_a_real_provider"}, _noop_progress)

    def test_registry_provider_without_key_raises(self):
        with self.assertRaises(ValueError):
            run_generate_teacher_data_task({"topic": "x", "provider": "registry:"}, _noop_progress)

    def test_unconfigured_anthropic_provider_raises_honest_error(self):
        # No ANTHROPIC_API_KEY set -- must fail clearly, not crash trying to
        # call an API with an empty key.
        with mock.patch("src.teacher_anthropic.AnthropicTeacher.check_connection",
                         return_value={"ok": False, "error": "ANTHROPIC_API_KEY not set"}):
            with self.assertRaises(RuntimeError) as ctx:
                run_generate_teacher_data_task({"topic": "photosynthesis"}, _noop_progress)
            self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_success_path_generates_and_stores_pairs(self):
        fake_pairs = [
            {"prompt": "What is X?", "ickle_answer": "", "improved_answer": "X is a thing.",
             "teacher_feedback": "", "score": 0.8, "tags": ["x"], "source_model": "anthropic"},
        ]
        with mock.patch("src.teacher_anthropic.AnthropicTeacher.check_connection",
                         return_value={"ok": True, "model": "claude-3-5-sonnet-20241022"}), \
             mock.patch("src.teacher_anthropic.AnthropicTeacher.batch_sft", return_value=fake_pairs) as mock_batch, \
             mock.patch("src.teacher_ingest.TeacherStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.start_session.return_value = mock.Mock(session_id="sess1")
            mock_store_instance.close_session.return_value = {}

            result = run_generate_teacher_data_task(
                {"topic": "photosynthesis", "count": 3, "provider": "anthropic"}, _noop_progress
            )

        mock_batch.assert_called_once_with("photosynthesis", count=3)
        self.assertEqual(result["turns_submitted"], 1)
        mock_store_instance.add_turn.assert_called_once()
        mock_store_instance.close_session.assert_called_once_with("sess1")

    def test_registry_provider_is_used_when_specified(self):
        with mock.patch("src.teacher_registry.RegistryTeacher.check_connection",
                         return_value={"ok": True, "provider": "openai:gpt-4o-mini"}), \
             mock.patch("src.teacher_registry.RegistryTeacher.batch_sft", return_value=[]) as mock_batch, \
             mock.patch("src.teacher_ingest.TeacherStore") as MockStore:
            mock_store_instance = MockStore.return_value
            mock_store_instance.start_session.return_value = mock.Mock(session_id="sess2")
            mock_store_instance.close_session.return_value = {}

            run_generate_teacher_data_task(
                {"topic": "x", "count": 2, "provider": "registry:openai:gpt-4o-mini"}, _noop_progress
            )
        mock_batch.assert_called_once_with("x", count=2)


class BuildDpoPrefsTaskTests(unittest.TestCase):
    """run_build_dpo_prefs_task wraps the already-existing, already-tested
    build_preference_pairs() -- these tests cover the new task's own
    validation (missing feedback file) and a real end-to-end pass against a
    real temp feedback file, since this function is cheap/pure-Python and a
    real run is more convincing than mocking it."""

    def test_missing_feedback_file_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = str(Path(tmp) / "does_not_exist.jsonl")
            with self.assertRaises(ValueError) as ctx:
                run_build_dpo_prefs_task({"feedback_path": missing_path}, _noop_progress)
            self.assertIn("rate a few", str(ctx.exception).lower())

    def test_builds_real_pairs_from_feedback_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            feedback_path = Path(tmp) / "hub_feedback.jsonl"
            out_path = Path(tmp) / "prefs.jsonl"
            rows = [
                {"prompt": "What is RoPE?", "response": "A good, detailed answer.", "rating": 5},
                {"prompt": "What is RoPE?", "response": "idk", "rating": 1},
            ]
            with feedback_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            result = run_build_dpo_prefs_task(
                {"feedback_path": str(feedback_path), "out_path": str(out_path)}, _noop_progress
            )

            self.assertEqual(result["pairs_written"], 1)
            self.assertTrue(out_path.exists())
            pair = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(pair["chosen"], "A good, detailed answer.")
            self.assertEqual(pair["rejected"], "idk")


class DpoTrainTaskTests(unittest.TestCase):
    """run_dpo_train_task wraps the already-implemented, already-tested
    run_dpo() -- covers the new task's own validation and argument wiring.
    The DPO training loop itself (run_dpo) is mocked here rather than run
    for real: it's a real torch training loop already covered by
    tests/test_dpo_train.py's lower-level helper tests, and re-running it
    end to end here would mostly be re-testing dpo_train.py, not this new
    wrapper."""

    def test_missing_prefs_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.jsonl")
            with self.assertRaises(ValueError):
                run_dpo_train_task({"prefs_path": missing, "model": "models/does_not_exist.pt"}, _noop_progress)

    def test_missing_model_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefs_path = Path(tmp) / "prefs.jsonl"
            prefs_path.write_text('{"prompt": "a", "chosen": "b", "rejected": "c"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                run_dpo_train_task({"prefs_path": str(prefs_path)}, _noop_progress)

    def test_success_path_calls_run_dpo_with_expected_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefs_path = Path(tmp) / "prefs.jsonl"
            prefs_path.write_text('{"prompt": "a", "chosen": "b", "rejected": "c"}\n', encoding="utf-8")
            out_model = str(Path(tmp) / "aligned.pt")

            with mock.patch("src.dpo_train.run_dpo", return_value={"sample_count": 1, "history": []}) as mock_run:
                result = run_dpo_train_task(
                    {
                        "prefs_path": str(prefs_path),
                        "model": "models/base.pt",
                        "out_model": out_model,
                        "steps": 5,
                    },
                    _noop_progress,
                )

        mock_run.assert_called_once_with(
            model_path="models/base.pt",
            preference_data_path=str(prefs_path),
            out_path=out_model,
            steps=5,
        )
        self.assertEqual(result["out_model"], out_model)
        self.assertEqual(result["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
