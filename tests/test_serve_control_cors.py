import unittest
import urllib.request

from src.serve_control import create_server, shutdown_server


class ControlServerCorsTests(unittest.TestCase):
    """Regression tests for cross-origin access to the control API.

    The desktop app's chat page is served by serve_web.py on one port and
    calls this server's API (serve_control.py) on a different port for the
    Manage panel -- a different port is a different browser origin, so
    without CORS headers every fetch() from that page is silently blocked
    even though the server itself responds fine. curl/Invoke-RestMethod
    don't enforce CORS, so a plain endpoint check doesn't catch this --
    these tests check the actual response headers a browser would evaluate.
    """

    def setUp(self):
        self.server = create_server(host="127.0.0.1", port=0, web_root="web")
        self.port = self.server.server_port
        import threading
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        shutdown_server(self.server)
        self.thread.join(timeout=5)

    def test_get_response_reflects_localhost_origin(self):
        # A different port (as the desktop app's chat page is) is a
        # different browser origin, so this simulates that legitimate
        # cross-port case rather than a bare wildcard.
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/tasks",
            headers={"Origin": "http://127.0.0.1:8787"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8787")

    def test_options_preflight_succeeds(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/tasks",
            method="OPTIONS",
            headers={"Origin": "http://127.0.0.1:8787"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertIn(resp.status, (200, 204))
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8787")
            self.assertIn("POST", resp.headers.get("Access-Control-Allow-Methods", ""))

    def test_non_local_origin_is_not_granted_cors_access(self):
        """Regression test: this used to be a bare `Access-Control-Allow-Origin: *`,
        which let any webpage a user had open -- not just Ickle's own UI --
        read this API's responses via a background fetch(). A random
        external site's Origin must not get the header at all."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/tasks",
            headers={"Origin": "https://evil.example.com"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))

    def test_request_without_origin_header_gets_no_cors_header(self):
        # curl/Invoke-RestMethod-style same-machine callers don't send an
        # Origin header at all and don't need one -- CORS only matters to
        # browsers evaluating a cross-origin fetch().
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/tasks")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertIsNone(resp.headers.get("Access-Control-Allow-Origin"))


if __name__ == "__main__":
    unittest.main()
