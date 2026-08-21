"""Ickle desktop app: chat *and* management (training, tasks) in one native
window instead of a browser tab plus a separate DearPyGui process.

This is a convenience wrapper for end users who want a taskbar app rather
than an ordinary browser tab. As of the control-API auto-start added to
serve_web.py, plain `serve-web` already gets the full Manage panel too (the
page auto-discovers the control port via /api/control-port instead of
needing it in the URL) -- this file no longer needs to orchestrate anything
special, it just runs that same self-sufficient chat server in a background
thread and opens it in a native OS window via pywebview. The `?control_port=`
URL param is still passed for one release as a belt-and-braces fallback in
case auto-discovery ever fails, but /api/control-port is the primary path.
"""
from __future__ import annotations

import argparse
import threading

from src.serve_web import create_server, find_free_port, shutdown_server, start_embedded_control_server
from src.serve_control import shutdown_server as shutdown_control_server


def _find_free_port(preferred: int) -> int:
    """Kept as a thin alias -- tests import this name directly."""
    return find_free_port(preferred)


def main():
    parser = argparse.ArgumentParser(description="Ickle desktop app (chat + management in a native window)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--control-port", type=int, default=8788)
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--width", type=int, default=1180)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    port = find_free_port(args.port)
    server = create_server(host="127.0.0.1", port=port, web_root=args.web_root)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    control_server, control_port = start_embedded_control_server(
        host="127.0.0.1", preferred_port=args.control_port, web_root=args.web_root
    )
    server.control_port = control_port  # type: ignore[attr-defined]

    import webview  # heavy GUI import kept lazy so `--help` and tests stay fast

    window = webview.create_window(
        "Ickle",
        f"http://127.0.0.1:{port}/?control_port={control_port}",
        width=args.width,
        height=args.height,
        min_size=(720, 480),
    )

    def _on_closed():
        shutdown_server(server)
        shutdown_control_server(control_server)

    window.events.closed += _on_closed
    webview.start()


if __name__ == "__main__":
    main()
