import importlib
import unittest
from unittest import mock


class PolicyIntegrationTests(unittest.TestCase):
    def test_system_limits_reads_policy_defaults(self):
        import src.system_limits as system_limits

        with mock.patch(
            "src.policy_loader.load_policy_safe",
            return_value={
                "limits": {
                    "max_tool_calls": 3,
                    "max_web_chars": 1200,
                    "web_timeout_ms": 9000,
                    "torch_threads": 2,
                    "require_clarification_on_vague": False,
                }
            },
        ):
            importlib.reload(system_limits)
            limits = system_limits.SystemLimits()
            self.assertEqual(limits.max_tool_calls, 3)
            self.assertEqual(limits.max_web_chars, 1200)
            self.assertEqual(limits.web_timeout_ms, 9000)
            self.assertEqual(limits.torch_threads, 2)
            self.assertFalse(limits.require_clarification_on_vague)

        importlib.reload(system_limits)

    def test_autonomy_reads_policy_defaults(self):
        import src.autonomy as autonomy

        with mock.patch(
            "src.policy_loader.load_policy_safe",
            return_value={"autonomy": {"default_mode": "direct"}},
        ):
            importlib.reload(autonomy)
            self.assertEqual(autonomy.default_mode_name(), "direct")

        importlib.reload(autonomy)


if __name__ == "__main__":
    unittest.main()
