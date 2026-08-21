import tempfile
import unittest

from src.runtime_flags import RuntimeFlagsStore


class RuntimeFlagsTests(unittest.TestCase):
    def test_current_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/runtime_flags.json"
            store = RuntimeFlagsStore(path=path)
            updated = store.update_flags({"current_model": "models/ickle_demo.pt"})
            self.assertEqual(updated.get("current_model"), "models/ickle_demo.pt")
            self.assertEqual(store.get_flags().get("current_model"), "models/ickle_demo.pt")

    def test_unknown_keys_are_ignored_by_update(self):
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/runtime_flags.json"
            store = RuntimeFlagsStore(path=path)
            store.update_flags({"not_a_flag": True})
            self.assertNotIn("not_a_flag", store.get_flags())


if __name__ == "__main__":
    unittest.main()
