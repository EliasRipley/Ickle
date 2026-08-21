import io
import threading
import time
import unittest
from unittest import mock
from urllib.error import HTTPError

from src.task_actions import (
    _default_stream_max_chars,
    _detect_no_steps_executed,
    _looks_useful_fact,
    _request_json,
    _start_lease_heartbeat,
    _wiki_url_from_title,
    infer_task_from_instruction,
    run_task,
)


class LeaseHeartbeatTests(unittest.TestCase):
    """Regression coverage for a real live incident: a training task streaming
    a large remote dataset produced no "significant" log line (and therefore
    no task-queue lease renewal) for longer than WORKER_LEASE_SECONDS while
    genuinely still working, which would falsely mark it "failed: worker
    lease expired" on the next server restart."""

    def test_calls_progress_repeatedly_without_any_subprocess_output(self):
        calls: list[str] = []
        lock = threading.Lock()

        def progress(msg: str):
            with lock:
                calls.append(msg)

        stop_event = _start_lease_heartbeat(progress, lambda: "still running", interval_seconds=0.02)
        try:
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with lock:
                    if len(calls) >= 3:
                        break
                time.sleep(0.01)
        finally:
            stop_event.set()

        with lock:
            self.assertGreaterEqual(len(calls), 3)
            self.assertTrue(all(c == "still running" for c in calls))

    def test_stops_promptly_once_stop_event_is_set(self):
        calls: list[str] = []
        stop_event = _start_lease_heartbeat(progress=calls.append, status_fn=lambda: "x", interval_seconds=0.02)
        time.sleep(0.05)
        stop_event.set()
        count_at_stop = len(calls)
        time.sleep(0.1)
        self.assertEqual(len(calls), count_at_stop)

    def test_a_raising_progress_callback_stops_the_heartbeat_instead_of_crashing(self):
        def progress(_msg: str):
            raise RuntimeError("cancelled")

        stop_event = _start_lease_heartbeat(progress, lambda: "x", interval_seconds=0.02)
        time.sleep(0.1)
        stop_event.set()  # no exception should have propagated out of the background thread


class DetectNoStepsExecutedTests(unittest.TestCase):
    """Regression coverage for a real incident: a 'retrain' meant to pick up
    a bug fix silently resumed a stale checkpoint from an unrelated earlier
    run at the same out_model path, saw it was already at/past the target
    step, and exited having trained nothing -- while task status still
    reported "completed", identical to a real successful run."""

    def test_detects_the_real_log_line(self):
        tail = ["Building optimizer...", "no training steps executed; checkpoint already at step=1199", "saved model: x.pt"]
        self.assertTrue(_detect_no_steps_executed(tail))

    def test_case_insensitive(self):
        tail = ["NO TRAINING STEPS EXECUTED; checkpoint already at step=5"]
        self.assertTrue(_detect_no_steps_executed(tail))

    def test_normal_successful_run_is_not_flagged(self):
        tail = ["step=1190 loss=2.1", "step=1199 train_loss=2.05", "saved model: x.pt"]
        self.assertFalse(_detect_no_steps_executed(tail))

    def test_empty_log_is_not_flagged(self):
        self.assertFalse(_detect_no_steps_executed([]))


class DefaultStreamMaxCharsTests(unittest.TestCase):
    def test_scales_up_with_more_steps(self):
        small = _default_stream_max_chars(steps=200, batch_size=22, block_size=256)
        large = _default_stream_max_chars(steps=10000, batch_size=22, block_size=256)
        self.assertLess(small, large)

    def test_never_below_the_previous_flat_floor(self):
        tiny = _default_stream_max_chars(steps=1, batch_size=1, block_size=1)
        self.assertGreaterEqual(tiny, 2_000_000)

    def test_bounded_by_a_ceiling_for_very_large_runs(self):
        huge = _default_stream_max_chars(steps=10_000_000, batch_size=64, block_size=2048)
        self.assertLessEqual(huge, 120_000_000)

    def test_falls_back_to_reasonable_batch_and_block_when_unset(self):
        # batch_size/block_size of 0 means "let the server auto-derive them" --
        # this must not divide by zero or collapse to the floor for a normal
        # step count, since that's exactly the bug being fixed (every run,
        # not just ones with explicit overrides, was capped at ~2MB).
        result = _default_stream_max_chars(steps=1200, batch_size=0, block_size=0)
        self.assertGreater(result, 2_000_000)

    def test_1200_steps_is_meaningfully_larger_than_the_old_flat_default(self):
        result = _default_stream_max_chars(steps=1200, batch_size=22, block_size=256)
        self.assertGreater(result, 2_000_000 * 2)


class TaskActionTests(unittest.TestCase):
    def test_request_json_retries_on_429_then_succeeds(self):
        err = HTTPError(
            url="https://example.test",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0"},
            fp=io.BytesIO(b"{}"),
        )
        resp = mock.MagicMock()
        resp.read.return_value = b'{"ok": true}'
        cm = mock.MagicMock()
        cm.__enter__.return_value = resp
        cm.__exit__.return_value = False
        with mock.patch("src.task_actions.urlopen", side_effect=[err, cm]) as patched_open, mock.patch(
            "src.task_actions.time.sleep"
        ) as patched_sleep:
            out = _request_json("https://example.test", timeout_sec=1, max_retries=2, base_backoff_sec=0.01)
        self.assertTrue(out["ok"])
        self.assertEqual(patched_open.call_count, 2)
        self.assertEqual(patched_sleep.call_count, 1)

    def test_request_json_does_not_retry_non_transient_http_error(self):
        err = HTTPError(
            url="https://example.test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b"{}"),
        )
        with mock.patch("src.task_actions.urlopen", side_effect=err), mock.patch(
            "src.task_actions.time.sleep"
        ) as patched_sleep:
            with self.assertRaises(HTTPError):
                _request_json("https://example.test", timeout_sec=1, max_retries=3)
        self.assertEqual(patched_sleep.call_count, 0)

    def test_infer_wikipedia_learning_task(self):
        task = infer_task_from_instruction(
            "Access Wikipedia and learn as much as you can about reinforcement learning."
        )
        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "learn_wikipedia_topic")
        self.assertIn("reinforcement learning", task["payload"]["topic"].lower())

    def test_infer_train_task(self):
        task = infer_task_from_instruction("Please retrain the model tonight.")
        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "train_model")

    def test_infer_wikipedia_with_training_sets_auto_pipeline(self):
        task = infer_task_from_instruction(
            "Access Wikipedia and learn as much as you can about English literature, then train yourself."
        )
        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "learn_wikipedia_topic")
        self.assertTrue(task["payload"]["auto_pipeline"])
        self.assertEqual(task["payload"]["topic"], "English literature")

    def test_infer_none_for_unrelated_text(self):
        task = infer_task_from_instruction("Hello there.")
        self.assertIsNone(task)

    def test_infer_research_notes_query(self):
        task = infer_task_from_instruction("Revisit research notes about English literature.")
        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "research_notes_query")
        self.assertIn("English literature", task["payload"]["query"])

    def test_infer_web_learning_task(self):
        task = infer_task_from_instruction("Research from the internet about mitochondria and then train yourself.")
        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "learn_web_topic")
        self.assertIn("mitochondria", task["payload"]["topic"].lower())
        self.assertTrue(task["payload"]["auto_pipeline"])

    def test_infer_web_learning_task_strips_internet_suffix_with_address_prefix(self):
        task = infer_task_from_instruction(
            "Ickle, learn as much as you can about coral reef bleaching and recovery from the internet."
        )
        self.assertIsNotNone(task)
        self.assertEqual(task["task_type"], "learn_web_topic")
        self.assertEqual(task["payload"]["topic"], "coral reef bleaching and recovery")
        self.assertFalse(task["payload"]["auto_pipeline"])

    def test_wiki_url_uses_underscores(self):
        self.assertEqual(
            _wiki_url_from_title("Probability theory"),
            "https://en.wikipedia.org/wiki/Probability_theory",
        )

    def test_noisy_wikipedia_navigation_text_not_counted_as_fact(self):
        noisy = "From Wikipedia, the free encyclopedia Look for Probability theory on one of Wikipedia's sister projects."
        self.assertFalse(_looks_useful_fact(noisy))

    def test_evaluate_model_task_uses_inline_quiz_and_passes(self):
        quiz = [
            {
                "title": "Event horizon",
                "question": "What is an event horizon?",
                "expected_fact": "An event horizon is the boundary around a black hole.",
                "keywords": ["event", "horizon", "boundary", "black", "hole"],
            },
            {
                "title": "General relativity",
                "question": "What is general relativity?",
                "expected_fact": "General relativity describes gravity as spacetime curvature.",
                "keywords": ["general", "relativity", "gravity", "spacetime", "curvature"],
            },
            {
                "title": "Black hole thermodynamics",
                "question": "What is black hole thermodynamics?",
                "expected_fact": "It relates black hole mechanics to thermodynamic laws.",
                "keywords": ["black", "hole", "thermodynamics", "laws"],
            },
        ]

        def chat_runner(payload):
            prompt = str(payload.get("prompt", "")).lower()
            model = str(payload.get("model", ""))
            if model == "candidate.pt":
                if "event horizon" in prompt:
                    return {"response": "An event horizon is the boundary around a black hole."}
                if "general relativity" in prompt:
                    return {"response": "General relativity says gravity is spacetime curvature."}
                return {"response": "Black hole thermodynamics links black holes with thermodynamic laws."}
            return {"response": "I am not sure."}

        result = run_task(
            task_type="evaluate_model",
            payload={
                "topic": "black holes",
                "candidate_model": "candidate.pt",
                "baseline_model": "baseline.pt",
                "quiz_items": quiz,
                "quiz_size": 3,
                "min_delta": 0.2,
                "min_candidate_avg": 0.5,
                "promote_if_pass": False,
            },
            progress=lambda _: None,
            memory_enabled=True,
            allow_auto_training_tasks=True,
            chat_runner=chat_runner,
        )
        self.assertTrue(result["passed"])
        self.assertGreater(result["candidate_avg_score"], result["baseline_avg_score"])

    def test_evaluate_model_task_can_require_model_only_pass(self):
        quiz = [
            {
                "title": "Event horizon",
                "question": "What is an event horizon?",
                "expected_fact": "An event horizon is the boundary around a black hole.",
                "keywords": ["event", "horizon", "boundary", "black", "hole"],
            },
            {
                "title": "General relativity",
                "question": "What is general relativity?",
                "expected_fact": "General relativity describes gravity as spacetime curvature.",
                "keywords": ["general", "relativity", "gravity", "spacetime", "curvature"],
            },
            {
                "title": "Black hole thermodynamics",
                "question": "What is black hole thermodynamics?",
                "expected_fact": "It relates black hole mechanics to thermodynamic laws.",
                "keywords": ["black", "hole", "thermodynamics", "laws"],
            },
        ]

        def chat_runner(payload):
            model = str(payload.get("model", ""))
            prompt = str(payload.get("prompt", "")).lower()
            memory_on = bool(payload.get("enable_memory", True))
            if memory_on and model == "candidate.pt":
                return {"response": "event horizon boundary black hole general relativity spacetime curvature thermodynamics laws"}
            if memory_on and model == "baseline.pt":
                return {"response": "event horizon boundary black hole"}
            return {"response": "I am not sure."}

        result = run_task(
            task_type="evaluate_model",
            payload={
                "topic": "black holes",
                "candidate_model": "candidate.pt",
                "baseline_model": "baseline.pt",
                "quiz_items": quiz,
                "quiz_size": 3,
                "min_delta": 0.0,
                "min_candidate_avg": 0.2,
                "require_model_only_pass": True,
                "model_only_min_delta": 0.0,
                "model_only_min_candidate_avg": 0.2,
                "promote_if_pass": False,
                "eval_enable_memory": True,
            },
            progress=lambda _: None,
            memory_enabled=True,
            allow_auto_training_tasks=True,
            chat_runner=chat_runner,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["model_only_pass"])

    def test_continual_guard_task_dispatches_and_returns_gate_result(self):
        fake_report = {
            "passed": True,
            "promoted": False,
            "candidate_model": "models/candidate.pt",
            "baseline_model": "models/base.pt",
            "promote_to": None,
            "scores": {"core_drop": 0.01, "new_gain": 0.02},
            "mixing": {"mixed_pairs": 1200},
        }
        with mock.patch("src.task_actions.run_guarded_step", return_value=fake_report) as patched:
            result = run_task(
                task_type="continual_guard_step",
                payload={
                    "baseline_model": "models/base.pt",
                    "out_model": "models/candidate.pt",
                    "promote_if_pass": False,
                },
                progress=lambda _: None,
                memory_enabled=True,
                allow_auto_training_tasks=True,
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate_model"], "models/candidate.pt")
        self.assertEqual(result["baseline_model"], "models/base.pt")
        self.assertAlmostEqual(result["core_drop"], 0.01, places=6)
        self.assertAlmostEqual(result["new_gain"], 0.02, places=6)
        self.assertTrue(patched.called)


if __name__ == "__main__":
    unittest.main()
