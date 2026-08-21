import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.serve_control import ControlRuntime
from src.serve_web import ChatRuntime


class ControlRuntimeChatDelegationTests(unittest.TestCase):
    """Regression coverage: ControlRuntime.run_chat used to be its own,
    independently-drifted copy of the args-building logic in serve_web.py's
    ChatRuntime.run_chat -- missing thinking_mode, agent_mode,
    allow_code_execution, and image_base64 support (everything added to the
    chat/agent path this session), and returning a narrower response shape
    (no confidence/think_assessment). It now delegates to a real ChatRuntime
    instance instead of maintaining a second copy."""

    def _runtime(self) -> ControlRuntime:
        runtime = ControlRuntime.__new__(ControlRuntime)  # avoid full __init__ (swarm, hardware probing, etc.)
        runtime._chat_runtime = ChatRuntime()
        return runtime

    def test_run_chat_forwards_agent_and_code_execution_flags(self):
        captured = {}

        def fake_generate_response(args):
            captured["agent_mode"] = args.agent_mode
            captured["allow_code_execution"] = args.allow_code_execution
            captured["thinking_mode"] = args.thinking_mode
            return {"response": "ok", "reasoning": "", "model": args.model}

        runtime = self._runtime()
        with mock.patch("src.serve_web._resolve_default_model", return_value="models/does_not_exist.pt"), \
             mock.patch("src.serve_web.generate_response", side_effect=fake_generate_response):
            result = runtime.run_chat({
                "prompt": "hi",
                "agent": True,
                "allow_code_execution": True,
                "thinking_mode": True,
                "model": "models/does_not_exist.pt",
            })

        self.assertEqual(captured, {"agent_mode": True, "allow_code_execution": True, "thinking_mode": True})
        self.assertEqual(result["response"], "ok")

    def test_run_chat_returns_full_response_shape(self):
        def fake_generate_response(args):
            return {
                "response": "hello",
                "reasoning": "because",
                "model": args.model,
                "confidence": 0.5,
                "think_assessment": "fine",
            }

        runtime = self._runtime()
        with mock.patch("src.serve_web._resolve_default_model", return_value="models/does_not_exist.pt"), \
             mock.patch("src.serve_web.generate_response", side_effect=fake_generate_response):
            result = runtime.run_chat({"prompt": "hi", "model": "models/does_not_exist.pt"})

        self.assertEqual(
            result,
            {
                "response": "hello",
                "reasoning": "because",
                "model": "models/does_not_exist.pt",
                "confidence": 0.5,
                "low_confidence": False,
                "think_assessment": "fine",
            },
        )

    def test_run_chat_still_respects_chat_disabled_flag(self):
        runtime = self._runtime()
        # Mock get_flags() rather than writing to the real, file-backed
        # data/runtime_flags.json -- that file is shared process-wide state,
        # and mutating it here (even temporarily) risks racing other tests
        # or leaving chat disabled for real if this test fails before a
        # cleanup step runs.
        with mock.patch.object(runtime._chat_runtime.flags, "get_flags", return_value={"chat_enabled": False}):
            with self.assertRaises(PermissionError):
                runtime.run_chat({"prompt": "hi"})


class ControlRuntimeTeacherStatsTests(unittest.TestCase):
    """Regression coverage for the Teach tab's stats display -- was a real
    gap before this session: no way to know how much teaching data existed
    without reading data/teacher/sessions.json by hand."""

    def _runtime(self) -> ControlRuntime:
        return ControlRuntime.__new__(ControlRuntime)  # avoid full __init__

    def test_sums_turn_count_across_sessions(self):
        runtime = self._runtime()
        fake_sessions = [
            {"session_id": "a", "turn_count": 3},
            {"session_id": "b", "turn_count": 5},
        ]
        with mock.patch("src.serve_control.TeacherStore") as MockStore:
            MockStore.return_value.list_sessions.return_value = fake_sessions
            stats = runtime.get_teacher_corpus_stats()
        self.assertEqual(stats, {"session_count": 2, "turn_count": 8})

    def test_no_sessions_reports_zero_not_an_error(self):
        runtime = self._runtime()
        with mock.patch("src.serve_control.TeacherStore") as MockStore:
            MockStore.return_value.list_sessions.return_value = []
            stats = runtime.get_teacher_corpus_stats()
        self.assertEqual(stats, {"session_count": 0, "turn_count": 0})


class ChatRuntimeListModelsTests(unittest.TestCase):
    """Regression coverage for a real bug: every training task (including
    the web UI's own "Start training" flow) writes its output to
    models/candidates/, but list_models() used to only glob models/
    directly -- a freshly, successfully trained model never appeared in the
    model picker at all. Confirmed live against a real completed training
    run before this fix."""

    def _in_temp_cwd(self, fn):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            os.chdir(td)
            try:
                fn()
            finally:
                os.chdir(cwd)

    def test_finds_models_in_both_root_and_candidates(self):
        def run():
            promoted = Path("models")
            candidates = promoted / "candidates"
            candidates.mkdir(parents=True, exist_ok=True)
            (promoted / "promoted.pt").write_text("a", encoding="utf-8")
            (candidates / "trained.pt").write_text("b", encoding="utf-8")

            runtime = ChatRuntime()
            rows = runtime.list_models(policy_only=False)
            names = {row["name"] for row in rows}
            self.assertEqual(names, {"promoted.pt", "trained.pt"})

        self._in_temp_cwd(run)

    def test_candidate_model_can_be_the_active_policy_pick(self):
        def run():
            candidates = Path("models/candidates")
            candidates.mkdir(parents=True, exist_ok=True)
            trained = candidates / "ickle_baseline.pt"
            trained.write_text("trained", encoding="utf-8")

            runtime = ChatRuntime()
            rows = runtime.list_models(policy_only=True)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "ickle_baseline.pt")
            self.assertEqual(rows[0]["policy_tag"], "active")

        self._in_temp_cwd(run)


class ControlRuntimeListModelsDelegationTests(unittest.TestCase):
    """ControlRuntime.list_models() used to be a byte-for-byte duplicate of
    ChatRuntime.list_models() (same bug, fixed independently in each --
    exactly the kind of drift this project has been actively removing). It
    now delegates instead."""

    def test_delegates_to_chat_runtime(self):
        runtime = ControlRuntime.__new__(ControlRuntime)
        runtime._chat_runtime = mock.Mock()
        runtime._chat_runtime.list_models.return_value = [{"name": "x.pt"}]

        result = runtime.list_models(limit=5, include_checkpoints=True, policy_only=False)

        runtime._chat_runtime.list_models.assert_called_once_with(
            limit=5, include_checkpoints=True, policy_only=False
        )
        self.assertEqual(result, [{"name": "x.pt"}])


if __name__ == "__main__":
    unittest.main()
