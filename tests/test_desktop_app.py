import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.desktop_app import _find_free_port
from src.serve_web import ChatRuntime, create_server, resolve_web_root, shutdown_server


class DesktopAppTests(unittest.TestCase):
    def test_find_free_port_returns_preferred_when_available(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        self.assertEqual(_find_free_port(free_port), free_port)

    def test_find_free_port_falls_back_when_preferred_is_taken(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            taken_port = holder.getsockname()[1]
            found = _find_free_port(taken_port)
            self.assertNotEqual(found, taken_port)
            self.assertGreater(found, 0)

    def test_create_server_reused_by_desktop_app_starts_and_stops(self):
        server = create_server(host="127.0.0.1", port=0, web_root="web")
        try:
            self.assertGreater(server.server_port, 0)
        finally:
            shutdown_server(server)

    def test_resolve_web_root_falls_back_to_app_root_when_cwd_lacks_it(self):
        # Regression test: a packaged .exe invoked from a directory that
        # doesn't have web/ alongside it (e.g. dist/ickle_client/, where the
        # bundled web/ actually lives under _internal/) used to fail with
        # "Web root not found" even though the assets were bundled correctly.
        # Uses a folder name that certainly doesn't exist relative to the
        # real cwd, so the app-root fallback triggers without needing to
        # fake the process's actual working directory.
        marker = "ickle_test_bundled_web_xyz"
        with tempfile.TemporaryDirectory() as fake_app_root:
            bundled_web = Path(fake_app_root) / marker
            bundled_web.mkdir()
            (bundled_web / "index.html").write_text("<html></html>", encoding="utf-8")

            with mock.patch("src.workspace_paths.get_app_root", return_value=Path(fake_app_root)):
                resolved = resolve_web_root(marker)
        self.assertEqual(resolved, bundled_web.resolve())

    def test_resolve_web_root_prefers_cwd_when_present(self):
        # Existing dev-checkout behavior must be unchanged: if web/ already
        # exists relative to cwd, use that -- don't prefer the app root.
        resolved = resolve_web_root("web")
        self.assertTrue(resolved.exists())
        self.assertEqual(resolved.name, "web")

    def test_chat_runtime_does_not_crash_with_no_models_directory(self):
        # Regression test: a fresh install/first run with no models/
        # directory yet (the normal state before a user has trained or
        # imported a model) used to crash ChatRuntime's constructor with an
        # unhandled FileNotFoundError -- taking the whole server down before
        # it could even start, let alone tell the user "no model yet".
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as empty_dir:
            try:
                os.chdir(empty_dir)
                runtime = ChatRuntime()
                self.assertEqual(runtime.default_model, "")
                status = runtime.get_status()
                self.assertEqual(status["chat_model"], "")
                self.assertEqual(status["model"]["size_bytes"], 0)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
