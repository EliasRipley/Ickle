import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from src.serve_web import create_server, shutdown_server


class FeedbackEndpointTests(unittest.TestCase):
    """Regression coverage for the web app's learn-from-chat pathway: before
    this, data/hub_feedback.jsonl (the only input build_feedback_corpus.py and
    build_preference_pairs.py read) could only be written by hub.py's REPL
    /feedback command -- and hub.py can't generate chat responses at all, so
    nothing a user did in the actual web app ever became training data."""

    def setUp(self):
        self.server = create_server(host="127.0.0.1", port=0, web_root="web")
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        shutdown_server(self.server)
        self.thread.join(timeout=5)

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_feedback_written_to_file_and_picked_up_by_corpus_builder(self):
        with tempfile.TemporaryDirectory() as d:
            feedback_path = Path(d) / "hub_feedback.jsonl"
            with mock.patch("src.serve_web.record_feedback") as mocked:
                from src.feedback_store import record_feedback as real_record_feedback

                mocked.side_effect = lambda **kw: real_record_feedback(path=str(feedback_path), **kw)

                status, body = self._post(
                    "/api/feedback",
                    {"prompt": "What is 2+2?", "response": "2+2 is 4.", "rating": 5, "notes": "correct"},
                )
            self.assertEqual(status, 200)
            self.assertTrue(body["saved"])
            self.assertEqual(body["rating"], 5)

            self.assertTrue(feedback_path.exists())
            lines = feedback_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["prompt"], "What is 2+2?")
            self.assertEqual(row["rating"], 5)

            from src.build_feedback_corpus import build_corpus

            out_path = Path(d) / "corpus.txt"
            build_corpus(str(feedback_path), str(out_path), min_rating=4)
            self.assertIn("What is 2+2?", out_path.read_text(encoding="utf-8"))

    def test_feedback_requires_prompt_and_response(self):
        status, body = self._post("/api/feedback", {"rating": 5})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_feedback_rejects_non_integer_rating(self):
        status, body = self._post(
            "/api/feedback", {"prompt": "hi", "response": "hello", "rating": "not-a-number"}
        )
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
