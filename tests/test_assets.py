import unittest
from pathlib import Path


class AssetTests(unittest.TestCase):
    def test_non_python_assets_exist(self):
        self.assertTrue(Path("api/ilm_openapi.yaml").exists())
        self.assertTrue(Path("web/index.html").exists())
        self.assertTrue(Path("web/app.js").exists())
        self.assertTrue(Path("web/styles.css").exists())
        self.assertTrue(Path("web/favicon.svg").exists())
        self.assertTrue(Path("data/maintenance/user_chat_benchmark.json").exists())
        self.assertTrue(Path("docs/INDEX.md").exists())

    def test_openapi_has_reality_path(self):
        txt = Path("api/ilm_openapi.yaml").read_text(encoding="utf-8")
        self.assertIn("/reality-check", txt)

    def test_openapi_has_epistemic_commons_paths(self):
        txt = Path("api/ilm_openapi.yaml").read_text(encoding="utf-8")
        self.assertIn("/api/epistemics/reviews", txt)
        self.assertIn("/api/commons/sync", txt)
        self.assertIn("/api/commons/adopt", txt)
        self.assertIn("/api/swarm/feedback", txt)

    def test_public_swarm_assets_and_api_exist(self):
        self.assertTrue(Path("docs/PUBLIC_SWARM.md").exists())
        self.assertTrue(Path("src/federated/public_dht.py").exists())
        txt = Path("api/ilm_openapi.yaml").read_text(encoding="utf-8")
        self.assertIn("/api/swarm/refresh", txt)

    def test_ui_uses_stable_capabilities_and_control_room_layout(self):
        html = Path("web/index.html").read_text(encoding="utf-8")
        css = Path("web/styles.css").read_text(encoding="utf-8")
        script = Path("web/app.js").read_text(encoding="utf-8")
        self.assertIn('id="capabilities-panel"', html)
        self.assertIn('id="agent-capabilities"', html)
        self.assertIn('class="network-status-board"', html)
        self.assertNotIn("There's no automatic public directory yet", html)
        self.assertIn("grid-template-columns: 224px minmax(0, 1fr)", css)
        self.assertIn("BitTorrent DHT", html)
        self.assertIn("/api/swarm/refresh", script)


if __name__ == "__main__":
    unittest.main()
