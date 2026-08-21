import unittest
from unittest import mock

from src.ilm_profile import detect_resources, resolve_resource_config, ResourceConfig


class ResourceConfigTests(unittest.TestCase):
    def test_detect_resources_returns_config(self):
        rc = detect_resources()
        self.assertIsInstance(rc, ResourceConfig)
        self.assertGreater(rc.cpu_cores, 0)
        self.assertGreater(rc.ram_gb, 0)

    def test_high_cpu_yields_larger_model(self):
        # RAM is pinned to a moderate, fixed value here so this test isolates
        # the effect of CPU count. Leaving real RAM detection in (via psutil)
        # let this test pass for the wrong reason for a long time: on a
        # machine with a lot of RAM, n_embd saturates its 768 cap regardless
        # of cpu_count, so both branches produced identical, ceiling-hit
        # values. That only stayed hidden because psutil wasn't actually
        # installed in this environment (silent ImportError fallback to a
        # hardcoded ram_gb=2.0 stub) -- fixed as part of the same session
        # that installed psutil for real hardware sizing.
        fake_mem = mock.Mock()
        fake_mem.total = 8 * (1024 ** 3)
        with mock.patch("psutil.virtual_memory", return_value=fake_mem):
            with mock.patch("src.ilm_profile.os.cpu_count", return_value=20):
                big = detect_resources()
            with mock.patch("src.ilm_profile.os.cpu_count", return_value=2):
                small = detect_resources()
        self.assertGreater(big.block_size, small.block_size)
        self.assertGreater(big.n_layer, small.n_layer)
        self.assertGreater(big.n_embd, small.n_embd)

    def test_percentage_overrides_work(self):
        from types import SimpleNamespace
        args = SimpleNamespace(cpu_pct=10, ram_pct=10, gpu_pct=0)
        rc = resolve_resource_config(args)
        self.assertEqual(rc.cpu_percent, 10)
        self.assertEqual(rc.ram_percent, 10)
        self.assertEqual(rc.gpu_percent, 0)

    def test_default_80_percent(self):
        rc = detect_resources()
        self.assertEqual(rc.cpu_percent, 80)
        self.assertEqual(rc.ram_percent, 80)
        self.assertEqual(rc.gpu_percent, 80)

    def test_derived_config_is_sane(self):
        rc = detect_resources()
        self.assertGreaterEqual(rc.block_size, 256)
        self.assertGreaterEqual(rc.n_embd, 64)
        self.assertGreaterEqual(rc.n_layer, 2)
        self.assertGreaterEqual(rc.n_head, 2)
        self.assertGreaterEqual(rc.batch_size, 4)
        self.assertGreaterEqual(rc.torch_threads, 1)


if __name__ == "__main__":
    unittest.main()
