import threading
import unittest
import urllib.request
import json

from src.serve_web import create_server, shutdown_server, start_embedded_control_server
from src.serve_control import shutdown_server as shutdown_control_server


class ControlPortDiscoveryTests(unittest.TestCase):
    """web/app.js discovers the control API port via GET /api/control-port
    instead of requiring ?control_port= in the URL (previously only
    desktop_app.py ever set that param, so a plain `serve-web` browser tab
    had no way to reach the Manage panel at all)."""

    def setUp(self):
        self.server = create_server(host="127.0.0.1", port=0, web_root="web")
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        shutdown_server(self.server)
        self.thread.join(timeout=5)

    def _get_control_port(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/control-port")
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read())

    def test_control_port_is_none_when_not_started(self):
        body = self._get_control_port()
        self.assertIsNone(body["control_port"])

    def test_control_port_reports_started_embedded_server(self):
        control_server, control_port = start_embedded_control_server(
            host="127.0.0.1", preferred_port=0, web_root="web"
        )
        self.server.control_port = control_port
        try:
            body = self._get_control_port()
            self.assertEqual(body["control_port"], control_port)
        finally:
            shutdown_control_server(control_server)


if __name__ == "__main__":
    unittest.main()
