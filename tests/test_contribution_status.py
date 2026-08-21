import threading
import unittest
import urllib.request
import json

from src.serve_control import create_server, shutdown_server


class ContributionStatusTests(unittest.TestCase):
    """The Sharing tab's contribution-ledger numbers (seed:peer ratio, rounds
    contributed) come from GET /api/contribution/status. Regression coverage
    for the endpoint existing and returning the shape the web UI expects --
    added alongside the existing /api/federated/status this mirrors."""

    def setUp(self):
        self.server = create_server(host="127.0.0.1", port=0, web_root="web")
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        shutdown_server(self.server)
        self.thread.join(timeout=5)

    def test_contribution_status_returns_expected_shape(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/contribution/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read())
        for key in (
            "seed_pieces_served",
            "seed_training_rounds",
            "peer_requests_served",
            "peer_requests_consumed",
            "contributed_score",
            "consumed_score",
            "ratio",
            "ratio_display",
            "has_history",
        ):
            self.assertIn(key, body)
        self.assertIsInstance(body["has_history"], bool)


if __name__ == "__main__":
    unittest.main()
