import tempfile
import unittest

from src.autonomy import get_mode
from src.capabilities import check_capability
from src.ilm_profile import detect_resources, resolve_resource_config, ResourceConfig
from src.state_store import ILMStateStore


class CoreBehaviorTests(unittest.TestCase):
    def test_capability_honesty_timer(self):
        report = check_capability("timer")
        self.assertTrue(report.supported)
        self.assertIn("set", report.summary.lower())

    def test_autonomy_mode_exists(self):
        mode = get_mode("power-user")
        self.assertEqual(mode.name, "power-user")

    def test_resource_config_detection(self):
        rc = detect_resources()
        self.assertIsInstance(rc, ResourceConfig)
        self.assertGreater(rc.cpu_cores, 0)
        self.assertGreater(rc.ram_gb, 0)
        self.assertGreaterEqual(rc.block_size, 256)
        self.assertGreaterEqual(rc.n_embd, 64)
        self.assertGreaterEqual(rc.n_layer, 2)

    def test_state_store_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/state.json"
            store = ILMStateStore(path)
            store.set_preference("tone", "direct")
            store.add_improvement_note("ask clarify first")

            store2 = ILMStateStore(path)
            self.assertEqual(store2.get_preference("tone"), "direct")
            self.assertEqual(len(store2.list_improvements()), 1)


if __name__ == "__main__":
    unittest.main()
