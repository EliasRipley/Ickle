"""Start Ickle's web UI (if not already running) and open it in a browser.

Replaces the previous two-step start_web_ui.bat flow (blind `start ""` of
serve-web, then a separate opener hardcoded to port 8787) -- that combo had
a real bug: serve-web's find_free_port() silently falls back to a random OS
port if 8787 is already taken (e.g. a leftover instance from an earlier
session), but the opener always checked port 8787 regardless. If that port
happened to be held by an old, stale, or unhealthy process, the browser
opened *that* instead of the freshly started instance -- indistinguishable
from "the app is showing an old/broken build" from the user's side, even
though a fresh instance really did start successfully on a different port.

This script closes that gap: reuse a healthy instance already on 8787 if
one exists AND it's actually running current code, otherwise start a new
one specifically on 8787 (not "whichever port happens to be free") so the
URL this script opens is always the server it's actually pointing at.

Staleness check: Python doesn't hot-reload edited .py/.js files, so an
already-running process from before a code change silently keeps serving
the old behavior forever with no visible sign anything is wrong -- that's
exactly the "I fixed the bug but you still see it" trap. /api/status
reports the process's own start time (ChatRuntime.started_at); if any file
under src/ or web/ was modified after that, the running process predates
the current code and gets replaced rather than reused.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8787
URL = f"http://127.0.0.1:{PORT}"


def _get_status() -> dict | None:
    try:
        with urllib.request.urlopen(f"{URL}/api/status", timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _latest_source_mtime() -> float:
    """Newest modification time across everything the running server's
    behavior actually depends on -- src/**/*.py (imported once at process
    start) and web/* (served fresh per-request, but still part of "the
    current build" from the user's perspective)."""
    latest = 0.0
    for pattern, base in (("**/*.py", ROOT / "src"), ("**", ROOT / "web")):
        for path in base.glob(pattern):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
    return latest


def _kill_stale_server() -> bool:
    """Find and terminate whatever process is listening on PORT. Returns
    True if something was found and asked to stop."""
    try:
        import psutil
    except ImportError:
        print(
            f"A stale Ickle server appears to be running on port {PORT}, but "
            f"psutil isn't installed so it can't be replaced automatically. "
            f"Stop it manually (Task Manager / taskkill) and re-run this script."
        )
        return False

    killed = False
    for conn in psutil.net_connections(kind="inet"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == PORT and conn.pid:
            try:
                psutil.Process(conn.pid).terminate()
                killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return killed


def main() -> int:
    status = _get_status()
    if status is not None:
        started_at = float(status.get("started_at") or 0)
        source_mtime = _latest_source_mtime()
        if started_at and started_at < source_mtime:
            print(
                f"Ickle is running on {URL} but it's serving an older build "
                f"than what's on disk now -- replacing it with a fresh instance."
            )
            if _kill_stale_server():
                for _ in range(20):
                    if _get_status() is None:
                        break
                    time.sleep(0.25)
            status = None
        else:
            print(f"Ickle is already running at {URL} -- opening it.")
            webbrowser.open(URL)
            return 0

    print(f"Starting Ickle web UI on {URL} ...")
    subprocess.Popen(
        [sys.executable, "-m", "src.app", "serve-web", "--port", str(PORT)],
        cwd=str(ROOT),
    )

    for _ in range(60):
        if _get_status() is not None:
            webbrowser.open(URL)
            print(f"UI opened at {URL}")
            return 0
        time.sleep(0.5)

    print(
        f"Server did not become ready on {URL} within 30s -- it may have "
        f"failed to bind that port (check for another process already "
        f"using it) or is still loading. Check the Ickle console window, "
        f"then try opening {URL} manually."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
