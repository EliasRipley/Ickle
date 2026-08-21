import os
import tempfile
import threading
import unittest

from src.trainer_providers import ProviderConfig, ProviderRegistry


class TrainerProvidersDeadlockTests(unittest.TestCase):
    """Regression coverage for a real bug: add_provider()/remove_provider()
    acquired self._lock (a plain threading.Lock) and then called _save(),
    which acquires the *same* lock -- a guaranteed self-deadlock on every
    single call. Run each operation on a daemon thread with a hard join
    timeout so a regression back to a non-reentrant lock fails the test
    instead of hanging the suite forever."""

    def _run_with_timeout(self, fn, timeout=5.0):
        result: dict = {}

        def target():
            result["value"] = fn()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)
        self.assertFalse(t.is_alive(), "call did not return -- looks deadlocked")
        return result.get("value")

    def test_add_provider_does_not_deadlock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ProviderRegistry(
                registry_path=os.path.join(tmpdir, "providers.json"),
                ledger_path=os.path.join(tmpdir, "usage.jsonl"),
            )
            cfg = ProviderConfig(provider="openai", model="gpt-4o-mini", api_key_env="ICKLE_TEST_KEY")
            out = self._run_with_timeout(lambda: reg.add_provider(cfg))
            self.assertEqual(out["status"], "added")
            self.assertEqual(len(reg.list_providers()), 1)

    def test_remove_provider_does_not_deadlock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = ProviderRegistry(
                registry_path=os.path.join(tmpdir, "providers.json"),
                ledger_path=os.path.join(tmpdir, "usage.jsonl"),
            )
            cfg = ProviderConfig(provider="openai", model="gpt-4o-mini", api_key_env="ICKLE_TEST_KEY")
            reg.add_provider(cfg)
            removed = self._run_with_timeout(lambda: reg.remove_provider("openai:gpt-4o-mini"))
            self.assertTrue(removed)
            self.assertEqual(len(reg.list_providers()), 0)


if __name__ == "__main__":
    unittest.main()
