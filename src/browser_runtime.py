"""Locate and launch a browser for Playwright-backed local web tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


WINDOWS_CHROMIUM_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
)


def launch_headless_browser(playwright: Any, *, headless: bool = True):
    """Launch a managed Playwright browser or a compatible system Chromium.

    Playwright's Python package can be installed while its separate browser
    download is absent. On Windows, Edge or Chrome is a useful local fallback.
    The return value is ``(browser, description)``.
    """
    configured = str(os.getenv("ICKLE_BROWSER_EXECUTABLE", "")).strip()
    if configured:
        executable = Path(configured).expanduser()
        if not executable.is_file():
            raise RuntimeError(f"ICKLE_BROWSER_EXECUTABLE does not exist: {executable}")
        browser = playwright.chromium.launch(headless=headless, executable_path=str(executable))
        return browser, f"configured Chromium ({executable.name})"

    firefox_executable = Path(playwright.firefox.executable_path)
    if firefox_executable.is_file():
        return playwright.firefox.launch(headless=headless), "Playwright Firefox"

    chromium_executable = Path(playwright.chromium.executable_path)
    if chromium_executable.is_file():
        return playwright.chromium.launch(headless=headless), "Playwright Chromium"

    if os.name == "nt":
        for executable in WINDOWS_CHROMIUM_CANDIDATES:
            if executable.is_file():
                browser = playwright.chromium.launch(headless=headless, executable_path=str(executable))
                return browser, f"system Chromium ({executable.name})"

    raise RuntimeError(
        "No Playwright-compatible browser is available. Run "
        "'python -m playwright install firefox' or set ICKLE_BROWSER_EXECUTABLE "
        "to a Chromium-based browser."
    )
