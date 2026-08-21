import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.browser_runtime import launch_headless_browser


class _BrowserType:
    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.calls = []

    def launch(self, **kwargs):
        self.calls.append(kwargs)
        return object()


class BrowserRuntimeTests(unittest.TestCase):
    def test_prefers_managed_firefox_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "firefox.exe"
            executable.write_bytes(b"test")
            firefox = _BrowserType(str(executable))
            chromium = _BrowserType(str(Path(tmp) / "missing.exe"))
            browser, description = launch_headless_browser(
                SimpleNamespace(firefox=firefox, chromium=chromium)
            )
        self.assertIsNotNone(browser)
        self.assertEqual(description, "Playwright Firefox")
        self.assertEqual(firefox.calls, [{"headless": True}])

    def test_configured_chromium_path_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "browser.exe"
            executable.write_bytes(b"test")
            firefox = _BrowserType(str(Path(tmp) / "missing-firefox.exe"))
            chromium = _BrowserType(str(Path(tmp) / "missing-chromium.exe"))
            with mock.patch.dict("os.environ", {"ICKLE_BROWSER_EXECUTABLE": str(executable)}):
                _, description = launch_headless_browser(
                    SimpleNamespace(firefox=firefox, chromium=chromium)
                )
        self.assertIn("configured Chromium", description)
        self.assertEqual(chromium.calls[0]["executable_path"], str(executable))


if __name__ == "__main__":
    unittest.main()
