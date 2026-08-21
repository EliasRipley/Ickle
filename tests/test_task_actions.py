import io
import unittest
from unittest import mock
from urllib.error import HTTPError

from src.task_actions import _looks_useful_fact, _request_json, _wiki_url_from_title, infer_task_from_instruction, run_task


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
