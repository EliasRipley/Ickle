"""Shared boilerplate for Ickle's two local http.server-based handlers
(serve_web.py's ChatHandler and serve_control.py's ControlHandler): no-cache
headers, optional scoped-CORS support, bounded JSON body parsing, JSON
response writing, and access-log gating. These used to be two independently
maintained copies that had already started drifting (e.g. only one of them
enforced a request-body size cap) -- this is the single implementation both
subclass instead.
"""

from __future__ import annotations

import json
import os
import re
from http.server import SimpleHTTPRequestHandler
from typing import Any

# 24 MiB accommodates a base64-encoded chat image attachment (up to 15 MiB
# raw, ~33% larger encoded) plus JSON overhead; every other endpoint on
# either server sends/receives payloads far smaller than this, so it's a
# shared ceiling rather than something each endpoint needs to tune.
DEFAULT_MAX_JSON_BYTES = 24 * 1024 * 1024

_LOCAL_ORIGIN_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", re.IGNORECASE)


def is_local_cors_origin(origin: str) -> bool:
    """True only for http(s)://127.0.0.1|localhost, any port -- the set of
    origins a same-machine Ickle UI page can legitimately be served from.
    Anything else (a random website, a remote host) gets no CORS header and
    stays blocked by the browser as normal."""
    return bool(origin) and bool(_LOCAL_ORIGIN_RE.match(origin.strip()))


class IckleHTTPHandler(SimpleHTTPRequestHandler):
    """Base class for Ickle's local API handlers.

    web_root: directory SimpleHTTPRequestHandler serves static files from.
    enable_cors: set True on a handler reachable cross-port (serve_control.py's
        Manage API, called from serve_web.py's chat page on a different
        port/origin) and left False where the caller is always same-origin
        (serve_web.py's own chat UI), since CORS headers are meaningless --
        and best not sent -- when nothing needs them.
    max_json_bytes: request-body ceiling for _read_json(); override per
        subclass if a handler genuinely needs a different bound.
    """

    web_root = "."
    enable_cors = False
    max_json_bytes = DEFAULT_MAX_JSON_BYTES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_root, **kwargs)

    def end_headers(self):
        # Avoid stale frontend bundles causing mismatched JS/UI behavior.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        if self.enable_cors:
            # A bare wildcard would let *any* webpage a user has open --
            # not just Ickle's own UI -- read this API's responses via a
            # background fetch(), since the server only binds 127.0.0.1 but
            # a wildcard Origin grants access to every page regardless of
            # its own origin. Reflecting the Origin back only when it's
            # another localhost port keeps the legitimate cross-port
            # desktop-app case working while a random external site gets no
            # CORS header at all and is blocked by the browser as usual.
            origin = self.headers.get("Origin", "")
            if is_local_cors_origin(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):  # noqa: N802
        # Browsers send a CORS preflight OPTIONS request before cross-origin
        # POSTs with a JSON body; without a handler this falls through to
        # SimpleHTTPRequestHandler's default (405/501), failing the
        # preflight and blocking every cross-origin POST. Harmless to
        # answer even when enable_cors is off (no CORS headers get added by
        # end_headers() in that case, so it changes nothing for same-origin
        # callers).
        self.send_response(204)
        self.end_headers()

    def _read_json(self) -> dict[str, Any]:
        raw_len = int(self.headers.get("Content-Length", "0"))
        if raw_len < 0 or raw_len > self.max_json_bytes:
            raise ValueError(f"JSON body exceeds the {self.max_json_bytes // (1024 * 1024)} MiB limit")
        raw = self.rfile.read(raw_len) if raw_len > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: int, payload: Any):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args):  # noqa: A002
        # Keep console focused on learning/training telemetry. Opt back in with:
        #   $env:ICKLE_HTTP_ACCESS_LOG='1'
        if str(os.getenv("ICKLE_HTTP_ACCESS_LOG", "0")).strip().lower() not in {"1", "true", "yes"}:
            return
        super().log_message(format, *args)
