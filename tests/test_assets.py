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


if __name__ == "__main__":
    unittest.main()
