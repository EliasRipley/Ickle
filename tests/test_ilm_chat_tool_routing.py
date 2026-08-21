import unittest
from types import SimpleNamespace
from unittest import mock

from src.ilm_chat import _maybe_route_local_tools, generate_response
from src.system_limits import SystemLimits


class _FakeAgent:
    def __init__(self, limits, autonomy_mode: str = "balanced"):
        self.limits = limits
        self.autonomy_mode = autonomy_mode

    def maybe_request_clarification(self, prompt: str):
        from src.clarify import ClarificationResult

        return ClarificationResult(needs_clarification=False)

    def timer_set(self, duration_text: str, name: str = "") -> str:
        return f"timer-set:{duration_text}:{name}"

    def timer_list(self) -> str:
        return "timer-list-ok"

    def timer_check(self, name: str) -> str:
        return f"timer-check:{name}"

    def timer_pause(self, name: str) -> str:
        return f"timer-pause:{name}"

    def timer_resume(self, name: str) -> str:
        return f"timer-resume:{name}"

    def timer_cancel(self, name: str) -> str:
        return f"timer-cancel:{name}"

    def desktop_control(self, action: str, payload_json: str = "{}") -> str:
        return f"desktop:{action}:{payload_json}"


class ILMChatToolRoutingTests(unittest.TestCase):
    def test_timer_command_routes_to_agent(self):
        limits = SystemLimits()
        with mock.patch("src.ilm_chat.LocalAgent", _FakeAgent):
            out = _maybe_route_local_tools("/timer-set 5 minutes|focus", limits)
        self.assertEqual(out, "timer-set:5 minutes:focus")

    def test_desktop_command_routes_to_agent(self):
        limits = SystemLimits()
        with mock.patch("src.ilm_chat.LocalAgent", _FakeAgent):
            out = _maybe_route_local_tools('/desktop click {"x":10,"y":20}', limits)
        self.assertEqual(out, 'desktop:click:{"x":10,"y":20}')

    def test_generate_response_short_circuits_model_for_tool_command(self):
        args = SimpleNamespace(
            model="models/unused.pt",
            prompt="/timer-list",
            max_new=64,
            max_new_limit=128,
            temperature=0.5,
            top_k=20,
            torch_threads=1,
            skill="",
            enable_memory=False,
            enable_web_tools=False,
            speculative=False,
            speculative_gamma=3,
        )
        with mock.patch("src.ilm_chat.LocalAgent", _FakeAgent), mock.patch(
            "src.ilm_chat._load_model_bundle",
            side_effect=AssertionError("model load should not happen for direct tool commands"),
        ):
            out = generate_response(args)
        self.assertEqual(out["response"], "timer-list-ok")


if __name__ == "__main__":
    unittest.main()

