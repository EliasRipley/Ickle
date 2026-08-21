import tempfile
import unittest
from pathlib import Path
from unittest import mock

import src.scoped_knowledge as scoped_knowledge_module
from src.serve_control import ControlRuntime


class DeltaThresholdEndpointTests(unittest.TestCase):
    """Regression coverage for the Add-ons tab's threshold control.
    DeltaRegistry.update_threshold() (src/delta_registry.py) was fully
    implemented but had no HTTP endpoint wiring it to anything -- the web UI
    had no way to call it at all."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        # get_scoped_manager() is a lazy module-level singleton keyed off
        # whatever data_dir the first caller in the process used; reset it so
        # this test gets its own isolated registry instead of whatever
        # another test (or another part of the app) already initialized.
        self._patcher = mock.patch.object(scoped_knowledge_module, "_scoped_manager", None)
        self._patcher.start()
        self.mgr = scoped_knowledge_module.get_scoped_manager(data_dir=self.tmpdir.name)
        self.mgr.register_delta(
            delta_id="test_delta",
            domain_description="testing",
            activation_threshold=0.6,
        )

    def tearDown(self):
        self._patcher.stop()
        self.tmpdir.cleanup()

    def test_set_threshold_updates_registry(self):
        runtime = ControlRuntime.__new__(ControlRuntime)  # avoid full __init__ (swarm, hardware probing, etc.)
        result = runtime.set_knowledge_delta_threshold("test_delta", 0.8)
        self.assertTrue(result["ok"])
        self.assertEqual(result["activation_threshold"], 0.8)

        entry = self.mgr.registry.get("test_delta")
        self.assertEqual(entry["activation_threshold"], 0.8)

    def test_set_threshold_clamps_out_of_range_values(self):
        runtime = ControlRuntime.__new__(ControlRuntime)
        result = runtime.set_knowledge_delta_threshold("test_delta", 5.0)
        self.assertEqual(result["activation_threshold"], 1.0)

        result = runtime.set_knowledge_delta_threshold("test_delta", -3.0)
        self.assertEqual(result["activation_threshold"], 0.0)

    def test_set_threshold_unknown_delta_fails_cleanly(self):
        runtime = ControlRuntime.__new__(ControlRuntime)
        result = runtime.set_knowledge_delta_threshold("does_not_exist", 0.5)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
