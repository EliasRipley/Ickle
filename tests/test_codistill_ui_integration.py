import json
import os
import tempfile
import unittest
from pathlib import Path


class CodistillStatusEndpointTests(unittest.TestCase):
    """get_codistill_status() is the data source for the Network tab's
    "Peer teaching" panel (web/app.js refreshCodistillPanel()) -- these pin
    its plain-dict shape so the two sides can't silently drift apart."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _status(self):
        # Imported lazily, inside the temp cwd, so ControlRuntime's own
        # __init__ side effects (swarm node, etc.) never touch the real
        # repo's data/ directory.
        from src.serve_control import ControlRuntime
        return ControlRuntime.get_codistill_status(object.__new__(ControlRuntime))

    def test_empty_state_has_no_report_or_trust(self):
        status = self._status()
        self.assertEqual(status["trust"], [])
        self.assertIsNone(status["last_report"])
        self.assertEqual(status["corpus_pairs"], 0)

    def test_reports_trust_ranking_and_last_report(self):
        Path("data/codistill").mkdir(parents=True)
        Path("data/codistill/peer_trust.json").write_text(
            json.dumps({"peer-a": 0.9, "peer-b": 0.2}), encoding="utf-8"
        )
        Path("data/codistill/last_report.json").write_text(
            json.dumps({"probes_taught": 3, "probes_total": 5, "peers_discovered": 2}),
            encoding="utf-8",
        )
        Path("data/codistill/distilled_corpus.txt").write_text(
            "User: a\nIckle: b\n\nUser: c\nIckle: d\n", encoding="utf-8"
        )

        status = self._status()
        self.assertEqual([row["peer_id"] for row in status["trust"]], ["peer-a", "peer-b"])
        self.assertEqual(status["last_report"]["probes_taught"], 3)
        self.assertEqual(status["corpus_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
