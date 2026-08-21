import unittest
from unittest import mock

from src.system_limits import SystemLimits, clamp_new_tokens


class SystemLimitsLiveReloadTests(unittest.TestCase):
    """Regression coverage for a real gap: SystemLimits' policy-driven field
    defaults used to be computed once at module-import time (a `_DEFAULTS =
    _load_limit_defaults()` module-level call baked into the dataclass field
    defaults). ilm_chat.py constructs a fresh SystemLimits() on every single
    chat turn, so a user editing config/ilm_policy.toml while Ickle is
    already running could never see the change take effect without a full
    process restart, even though a "fresh" limits object was being built
    every message. Defaults must now be re-read per instantiation."""

    def test_new_instance_picks_up_changed_policy_without_reimport(self):
        with mock.patch(
            "src.system_limits.load_policy_safe",
            return_value={"limits": {"max_tool_calls": 3}},
        ):
            limits = SystemLimits()
            self.assertEqual(limits.max_tool_calls, 3)

        with mock.patch(
            "src.system_limits.load_policy_safe",
            return_value={"limits": {"max_tool_calls": 99}},
        ):
            limits2 = SystemLimits()
            self.assertEqual(limits2.max_tool_calls, 99)

    def test_explicit_constructor_arg_overrides_policy(self):
        with mock.patch(
            "src.system_limits.load_policy_safe",
            return_value={"limits": {"torch_threads": 2}},
        ):
            limits = SystemLimits(torch_threads=16)
            self.assertEqual(limits.torch_threads, 16)

    def test_falls_back_to_hardcoded_defaults_without_policy_file(self):
        with mock.patch("src.system_limits.load_policy_safe", return_value={}):
            limits = SystemLimits()
            self.assertEqual(limits.max_tool_calls, 8)
            self.assertEqual(limits.max_web_chars, 5000)
            self.assertTrue(limits.require_clarification_on_vague)

    def test_clamp_new_tokens(self):
        self.assertEqual(clamp_new_tokens(500, 256), 256)
        self.assertEqual(clamp_new_tokens(10, 256), 10)


if __name__ == "__main__":
    unittest.main()
