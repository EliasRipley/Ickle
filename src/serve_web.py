"""
Ickle chat server — serves the web UI with chat-only endpoints.
For training, tasks, maintenance, swarm, etc., launch serve_control.py.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.chat_sessions import ChatSessions
from src.feedback_store import record_feedback
from src.friendly_errors import friendly_error_message
from src.ilm_chat import _resolve_default_model, generate_response
from src.ilm_chat_generation import (
    _detect_auto_torch_threads,
    _generate_model_response,
    _load_model_bundle,
    _resolve_default_model as _gen_resolve_default,
    generate_events_stream,
)
from src.icklization import ick
from src.generation_queue import GenerationQueue
from src.http_handler_base import IckleHTTPHandler
from src.ilm_memory import get_memory
from src.ilm_chat import (
    detect_web_request,
    _build_memory_context,
    _try_memory_write,
)
from src.dynamic_web_reader import read_url_dynamic
from src.epistemics import build_answer_map
from src.federated.knowledge_commons import EpistemicLedger
from src.reality_check import collect_checks
from src.runtime_flags import RuntimeFlagsStore
from src.system_limits import SystemLimits
from src.training_control import inspect_training_status
from src.workspace_paths import get_training_root

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MiB decoded; generous for a phone photo, bounded against abuse

# The image-attach thumbnail used to be shown via a blob: object URL, which
# WebView2 (the desktop app's rendering engine) has documented, unresolved
# quirks rendering (blank/broken image), the same as it does for base64
# data: URIs -- the one source WebView2 reliably displays is a plain http://
# URL, which the app is already loaded over. So the preview is served back
# through this same local server instead of being inlined into the DOM.
ATTACH_PREVIEW_TTL_S = 300
ATTACH_PREVIEW_MAX_ENTRIES = 5
ATTACH_PREVIEW_MAX_BYTES = 8 * 1024 * 1024


def _write_temp_image(image_base64: str) -> str:
    """Decode a base64 image (chat attachment) to a temp file for the vision
    tools (src/tools/image_reader.py) to read. Caller is responsible for
    deleting the returned path once done."""
    # Browsers send data URLs (data:image/png;base64,....) if the caller
    # forgot to strip the prefix -- strip it defensively rather than fail.
    if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid image data: {exc}") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit")
    fd, path = tempfile.mkstemp(suffix=".img", prefix="ickle_chat_image_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


class ChatRuntime:
    def __init__(self):
        self.flags = RuntimeFlagsStore()
        # Resolved lazily via _resolve_default_model() below, not here: on a
        # fresh install with no models/ directory yet (the normal state for
        # a just-packaged/just-installed app before the user has trained or
        # imported anything), calling the module-level _resolve_default_model()
        # unguarded here raised FileNotFoundError straight out of the
        # constructor -- crashing the whole server with a traceback before it
        # could even start and tell the user "no model yet" gracefully.
        self.default_model = ""
        self._attach_previews: dict[str, tuple[bytes, str, float]] = {}
        self._attach_previews_lock = threading.Lock()
        self._commons: EpistemicLedger | None = None
        # Exposed via /api/status so a launcher (scripts/start_web_ui.py) can
        # tell a genuinely fresh process apart from a stale one still serving
        # code from before the last edit -- Python doesn't hot-reload source
        # files, so reusing an old process silently serves old behavior with
        # no visible sign anything is wrong.
        self.started_at = time.time()

    def epistemic_ledger(self) -> EpistemicLedger:
        # Lazy so read-only status/model-list operations on a fresh install do
        # not create an identity/database until epistemic features are used.
        if self._commons is None:
            self._commons = EpistemicLedger()
        return self._commons

    def add_epistemic_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        claim_text = str(payload.get("claim_text", "")).strip()
        relation = str(payload.get("relation", "")).strip().lower()
        if not claim_text:
            raise ValueError("Missing claim_text")
        event = self.epistemic_ledger().add_review(
            claim_text=claim_text,
            relation=relation,
            correction_text=str(payload.get("correction_text", "")).strip(),
            source_url=str(payload.get("source_url", "")).strip(),
            shared=bool(payload.get("shared", False)),
        )
        return {"saved": True, "event": event, "commons": self.epistemic_ledger().summary()}

    def store_attach_preview(self, data: bytes, content_type: str) -> str:
        if len(data) > ATTACH_PREVIEW_MAX_BYTES:
            raise ValueError(f"Preview image exceeds the {ATTACH_PREVIEW_MAX_BYTES // (1024 * 1024)} MiB limit")
        now = time.time()
        with self._attach_previews_lock:
            for key, (_, _, expires_at) in list(self._attach_previews.items()):
                if expires_at < now:
                    del self._attach_previews[key]
            while len(self._attach_previews) >= ATTACH_PREVIEW_MAX_ENTRIES:
                oldest_key = min(self._attach_previews, key=lambda k: self._attach_previews[k][2])
                del self._attach_previews[oldest_key]
            preview_id = uuid.uuid4().hex
            self._attach_previews[preview_id] = (data, content_type or "application/octet-stream", now + ATTACH_PREVIEW_TTL_S)
        return preview_id

    def get_attach_preview(self, preview_id: str) -> tuple[bytes, str] | None:
        with self._attach_previews_lock:
            entry = self._attach_previews.get(preview_id)
            if not entry:
                return None
            data, content_type, expires_at = entry
            if expires_at < time.time():
                del self._attach_previews[preview_id]
                return None
            return data, content_type

    def delete_attach_preview(self, preview_id: str) -> None:
        with self._attach_previews_lock:
            self._attach_previews.pop(preview_id, None)

    def _resolve_default_model(self) -> str:
        try:
            self.default_model = _resolve_default_model()
        except FileNotFoundError:
            pass
        return self.default_model

    def run_chat(self, payload: dict) -> dict:
        flags = self.flags.get_flags()
        if not flags.get("chat_enabled", True):
            raise PermissionError("Chat is disabled by runtime flags.")
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Missing prompt")
        memory_enabled = (
            bool(payload["enable_memory"]) if "enable_memory" in payload else bool(flags.get("memory_enabled", True))
        )
        web_tools_enabled = (
            bool(payload["enable_web_tools"])
            if "enable_web_tools" in payload
            else bool(flags.get("web_tools_enabled", True))
        )
        default_model = self._resolve_default_model()

        # Only this device owner's explicit reviews can enter the prompt.
        # Imported peer perspectives remain inspectable but inert until the
        # owner adopts one through the Commons UI.
        epistemic_context = ""
        try:
            epistemic_context = self.epistemic_ledger().context_for_prompt(prompt)
        except Exception as exc:  # Optional transparency must not break chat.
            print(f"epistemic context unavailable, continuing without it: {exc}", file=sys.stderr)

        image_path = None
        image_base64 = str(payload.get("image_base64") or "").strip()
        if image_base64:
            image_path = _write_temp_image(image_base64)

        args = SimpleNamespace(
            model=str(payload.get("model") or default_model),
            prompt=prompt,
            max_new=int(payload.get("max_new", 260)),
            max_new_limit=int(payload.get("max_new_limit", 500)),
            temperature=float(payload.get("temperature", 0.25)),
            top_k=int(payload.get("top_k", 20)),
            torch_threads=int(payload.get("torch_threads", 4)),
            skill=str(payload.get("skill", "")),
            enable_memory=memory_enabled,
            enable_web_tools=web_tools_enabled,
            speculative=bool(payload.get("speculative", False)),
            speculative_gamma=max(1, int(payload.get("speculative_gamma", 3))),
            thinking_mode=bool(payload.get("thinking_mode", False)),
            thinking=bool(payload.get("thinking_mode", False)),
            think_budget=int(payload.get("think_budget", 0) or 0),
            agent=bool(payload.get("agent", False)),
            agent_mode=bool(payload.get("agent", False)),
            allow_code_execution=bool(payload.get("allow_code_execution", False)),
            autonomy_mode=str(payload.get("autonomy_mode") or "") or None,
            image_path=image_path,
            raw_output=bool(payload.get("raw_output", False)),
            epistemic_context=epistemic_context,
        )
        try:
            result = generate_response(args)
        finally:
            if image_path:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except OSError:
                    pass
        output = {
            "response": result.get("response", ""),
            "reasoning": result.get("reasoning", ""),
            "model": result.get("model", args.model),
            "confidence": result.get("confidence"),
            "low_confidence": bool(result.get("low_confidence", False)),
            "think_assessment": result.get("think_assessment"),
        }
        try:
            output["epistemics"] = build_answer_map(
                prompt=prompt,
                response=str(output["response"] or ""),
                evidence_items=list(result.get("evidence_items", []) or []),
                review_lookup=self.epistemic_ledger(),
                low_confidence=bool(output["low_confidence"]),
            )
        except Exception as exc:  # Keep the model's answer available even if the optional map fails.
            print(f"answer map unavailable, continuing without it: {exc}", file=sys.stderr)
            output["epistemics"] = None
        return output

    def list_models(self, *, limit: int = 80, include_checkpoints: bool = False, policy_only: bool = True) -> list[dict]:
        rows: list[dict] = []
        model_root = Path("models")
        if not model_root.exists():
            return rows
        # Every training task (the web UI's own "Start training" flow
        # included) writes its output to models/candidates/, not models/
        # directly -- a bare model_root.glob("*.pt") only looks at the given
        # directory, never subdirectories, so candidate models were
        # completely invisible here (couldn't be selected in the model
        # picker, and _resolve_default_model() couldn't find them either).
        # Same bug class already fixed once in model_maintain.py's cleanup
        # sweep; scanning both known locations explicitly (rather than a
        # blind rglob) matches that existing fix's approach.
        search_dirs = [model_root, model_root / "candidates"]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for p in search_dir.glob("*.pt"):
                if not p.is_file():
                    continue
                if not include_checkpoints and p.name.endswith(".checkpoint.pt"):
                    continue
                rows.append(
                    {
                        "name": p.name,
                        "path": str(p),
                        "size_bytes": int(p.stat().st_size),
                        "updated_at": p.stat().st_mtime,
                        "meta_exists": Path(str(p) + ".meta.json").exists(),
                    }
                )
        rows.sort(key=lambda row: row["updated_at"], reverse=True)
        total = len(rows)
        for idx, row in enumerate(rows):
            row["newest_rank"] = idx + 1
            row["version_number"] = max(1, total - idx)
            row["version_label"] = f"v{max(1, total - idx):04d}"

        if not policy_only:
            return rows[: max(1, limit)]

        active_resolved = str(Path(self._resolve_default_model()).resolve())
        selected: list[dict] = []
        seen: set[str] = set()

        for row in rows:
            try:
                resolved = str(Path(str(row["path"])).resolve())
            except Exception:
                resolved = str(row["path"])
            if resolved == active_resolved:
                tagged = dict(row)
                tagged["policy_tag"] = "active"
                selected.append(tagged)
                seen.add(resolved)
                break

        for row in rows:
            if len(selected) >= 3:
                break
            try:
                resolved = str(Path(str(row["path"])).resolve())
            except Exception:
                resolved = str(row["path"])
            if resolved in seen:
                continue
            tagged = dict(row)
            tagged["policy_tag"] = "recent"
            selected.append(tagged)
            seen.add(resolved)

        return selected[: max(1, min(limit, 3))]

    def get_status(self) -> dict:
        flags = self.flags.get_flags()
        default_model = self._resolve_default_model()
        model_path = Path(default_model)
        return {
            "flags": flags,
            "chat_model": default_model,
            "model": {
                "name": model_path.name,
                "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
            },
            "training": inspect_training_status(get_training_root() / "training_live.json"),
            "local_first": True,
            "reality_checks": [asdict(c) for c in collect_checks()],
            "started_at": self.started_at,
        }

    def update_flags(self, updates: dict) -> dict:
        if not isinstance(updates, dict):
            raise ValueError("Flag updates must be an object")
        return self.flags.update_flags(updates)


class ChatHandler(IckleHTTPHandler):
    web_root = "."
    # Same-origin only (the chat page is served from this same process/port),
    # so no CORS headers are needed here -- unlike serve_control.py's
    # ControlHandler, which the desktop app's chat page calls cross-port.
    enable_cors = False

    @property
    def runtime(self) -> ChatRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    @property
    def gen_queue(self) -> GenerationQueue:
        return self.server.gen_queue  # type: ignore[attr-defined]

    @property
    def sessions(self) -> ChatSessions:
        return self.server.sessions  # type: ignore[attr-defined]

    def _handle_chat_stream(self, query: dict, *, query_values_are_lists: bool = True):
        def value(name: str, default: Any = "") -> Any:
            raw = query.get(name, [default] if query_values_are_lists else default)
            if query_values_are_lists and isinstance(raw, list):
                return raw[0] if raw else default
            return raw

        prompt_raw = value("prompt", "")
        prompt = str(prompt_raw).strip()
        if not prompt:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Empty prompt"}')
            return
        default_model = self.runtime._resolve_default_model()
        model_path = str(value("model", default_model)).strip() or default_model
        thinking_mode = str(value("thinking_mode", "false") or "false").strip().lower() in {
            "1", "true", "yes",
        }
        session_id = str(value("session_id", "") or "").strip() or None
        enable_memory = str(value("enable_memory", "true")).strip().lower() in {
            "1", "true", "yes",
        }
        enable_web_tools = str(value("enable_web_tools", "true")).strip().lower() in {
            "1", "true", "yes",
        }
        agent_mode = str(value("agent", "false") or "false").strip().lower() in {
            "1", "true", "yes",
        }
        allow_code_execution = str(value("allow_code_execution", "false") or "false").strip().lower() in {
            "1", "true", "yes",
        }
        raw_output = str(value("raw_output", "false") or "false").strip().lower() in {
            "1", "true", "yes",
        }
        # Only meaningful on the POST/JSON-body variant of this endpoint --
        # a multi-MB base64 image in a query string would hit URL-length
        # limits long before it got here, so GET callers simply won't have it.
        image_base64 = str(value("image_base64", "") or "").strip()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Each request represents one finite generation. Closing after `done`
        # lets fetch/EventSource readers finish instead of waiting forever.
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        try:
            result = self.runtime.run_chat({
                "model": model_path,
                "prompt": prompt,
                "thinking_mode": thinking_mode,
                "enable_memory": enable_memory,
                "enable_web_tools": enable_web_tools,
                "agent": agent_mode,
                "allow_code_execution": allow_code_execution,
                "raw_output": raw_output,
                "image_base64": image_base64,
            })
            reasoning = str(result.get("reasoning", "") or "")
            if reasoning:
                for event in (
                    {"type": "reasoning_start"},
                    {"type": "reasoning", "text": reasoning},
                    {"type": "reasoning_end", "text": reasoning},
                ):
                    line = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
            response = str(result.get("response", "") or "")
            for offset in range(0, len(response), 48):
                event = {"type": "text", "text": response[offset : offset + 48]}
                line = f"event: text\ndata: {json.dumps(event)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()
            done_event = {
                "type": "done",
                "low_confidence": bool(result.get("low_confidence", False)),
                "confidence": result.get("confidence"),
                "epistemics": result.get("epistemics"),
            }
            self.wfile.write(f"event: done\ndata: {json.dumps(done_event)}\n\n".encode("utf-8"))
            self.wfile.flush()
        except ConnectionError:
            pass
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            err = {"type": "error", "text": friendly_error_message(exc)}
            try:
                self.wfile.write(f"event: stream_error\ndata: {json.dumps(err)}\n\n".encode())
                self.wfile.flush()
            except OSError:
                pass

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/api/chat/stream":
            self._handle_chat_stream(query)
            return
        if parsed.path == "/reality-check":
            self._send_json(200, [asdict(c) for c in collect_checks()])
            return
        if parsed.path == "/api/status":
            self._send_json(200, self.runtime.get_status())
            return
        if parsed.path == "/api/control-port":
            # Lets app.js auto-discover the management/control API instead of
            # requiring a `?control_port=` URL param (previously only
            # desktop_app.py ever set that param, so a plain `serve-web`
            # browser tab had no way to reach Manage/Training/Network/etc.
            # at all). None means this process didn't start one (--no-control).
            self._send_json(200, {"control_port": getattr(self.server, "control_port", None)})
            return
        if parsed.path == "/api/models":
            limit = int((query.get("limit", ["80"])[0] or "80"))
            include_checkpoints = str((query.get("include_checkpoints", ["0"])[0] or "0")).strip() in {
                "1", "true", "yes",
            }
            all_models = str((query.get("all", ["0"])[0] or "0")).strip() in {"1", "true", "yes"}
            self._send_json(
                200,
                {
                    "models": self.runtime.list_models(
                        limit=limit,
                        include_checkpoints=include_checkpoints,
                        policy_only=not all_models,
                    )
                },
            )
            return
        if parsed.path == "/api/flags":
            self._send_json(200, self.runtime.flags.get_flags())
            return
        if parsed.path == "/api/sessions":
            self._send_json(200, {"sessions": self.sessions.list_sessions()})
            return
        if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/messages"):
            session_id = parsed.path.split("/")[3]
            session = self.sessions.get_session(session_id)
            if session is None:
                self._send_json(404, {"error": "Session not found"})
            else:
                self._send_json(200, {"messages": session.get("messages", [])})
            return
        if parsed.path.startswith("/api/attach-preview/"):
            preview_id = parsed.path.split("/")[3]
            entry = self.runtime.get_attach_preview(preview_id)
            if entry is None:
                self._send_json(404, {"error": "Preview not found or expired"})
                return
            data, content_type = entry
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"error": friendly_error_message(exc)})
            return

        try:
            if parsed.path == "/api/chat/stream":
                self._handle_chat_stream(payload, query_values_are_lists=False)
                return
            if parsed.path == "/api/chat":
                if self.gen_queue:
                    result = self.gen_queue.enqueue(payload)
                else:
                    result = self.runtime.run_chat(payload)
                self._send_json(200, result)
                return
            if parsed.path == "/api/flags":
                self._send_json(200, self.runtime.update_flags(payload))
                return
            if parsed.path == "/api/feedback":
                prompt = str(payload.get("prompt", "")).strip()
                response = str(payload.get("response", "")).strip()
                if not prompt or not response:
                    self._send_json(400, {"error": "Missing prompt or response"})
                    return
                try:
                    rating = int(payload.get("rating", 0))
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "rating must be an integer 1-5"})
                    return
                fb = record_feedback(
                    prompt=prompt,
                    response=response,
                    rating=rating,
                    notes=str(payload.get("notes", "")),
                )
                self._send_json(200, {"saved": True, "rating": fb.rating})
                return
            if parsed.path == "/api/epistemics/reviews":
                self._send_json(201, self.runtime.add_epistemic_review(payload))
                return
            if parsed.path == "/api/attach-preview":
                image_base64 = str(payload.get("image_base64", "") or "").strip()
                if not image_base64:
                    self._send_json(400, {"error": "Missing image_base64"})
                    return
                if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
                    image_base64 = image_base64.split(",", 1)[1]
                try:
                    raw = base64.b64decode(image_base64, validate=True)
                except (binascii.Error, ValueError) as exc:
                    self._send_json(400, {"error": f"Invalid image data: {exc}"})
                    return
                content_type = str(payload.get("content_type", "") or "image/jpeg").strip()
                try:
                    preview_id = self.runtime.store_attach_preview(raw, content_type)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(201, {"id": preview_id, "url": f"/api/attach-preview/{preview_id}"})
                return
            if parsed.path == "/api/sessions":
                title = str(payload.get("title", "")).strip()
                session = self.sessions.create_session(title=title)
                self._send_json(201, session)
                return
            if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/messages"):
                session_id = parsed.path.split("/")[3]
                role = str(payload.get("role", "user")).strip()
                text = str(payload.get("text", "")).strip()
                if not text:
                    self._send_json(400, {"error": "Missing text"})
                    return
                msg = self.sessions.add_message(
                    session_id=session_id,
                    role=role,
                    text=text,
                    thinking=str(payload.get("thinking", "")),
                    model=str(payload.get("model", "")),
                    epistemics=payload.get("epistemics"),
                    low_confidence=bool(payload.get("low_confidence", False)),
                )
                if msg is None:
                    self._send_json(404, {"error": "Session not found"})
                else:
                    self._send_json(200, {"message": msg})
                return
            if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/delete"):
                session_id = parsed.path.split("/")[3]
                ok = self.sessions.delete_session(session_id)
                self._send_json(200 if ok else 404, {"deleted": ok})
                return
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._send_json(400, {"error": friendly_error_message(exc)})
            return

        self._send_json(404, {"error": "Not found"})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/sessions/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 3:
                session_id = parts[2]
                ok = self.sessions.delete_session(session_id)
                self._send_json(200 if ok else 404, {"deleted": ok})
                return
        if parsed.path.startswith("/api/attach-preview/"):
            preview_id = parsed.path.split("/")[3]
            self.runtime.delete_attach_preview(preview_id)
            self._send_json(200, {"deleted": True})
            return
        self._send_json(404, {"error": "Not found"})


def resolve_web_root(web_root: str) -> Path:
    """Resolve --web-root, falling back to the bundled app root if the plain
    cwd-relative path doesn't exist. A packaged .exe's data (see
    get_app_root()) doesn't live in the process's current working directory
    -- it lives wherever PyInstaller unpacked it -- so a bare
    Path(web_root).resolve() only works by coincidence in a dev checkout run
    from the project root."""
    candidate = Path(web_root)
    resolved_root = candidate.resolve()
    if not resolved_root.exists() and not candidate.is_absolute():
        from src.workspace_paths import get_app_root
        app_root_candidate = (get_app_root() / candidate).resolve()
        if app_root_candidate.exists():
            resolved_root = app_root_candidate
    return resolved_root


def create_server(
    host: str = "127.0.0.1",
    port: int = 8787,
    web_root: str = "web",
    no_batching: bool = False,
) -> ThreadingHTTPServer:
    """Build (but don't start) a ready-to-serve chat server. Shared by the
    CLI entry point below and src/desktop_app.py, which runs this same
    server in a background thread inside a native window process instead of
    requiring a separate `serve-web` process + manually opened browser tab."""
    resolved_root = resolve_web_root(web_root)
    if not resolved_root.exists():
        raise SystemExit(f"Web root not found: {resolved_root}")

    runtime = ChatRuntime()
    gen_queue = GenerationQueue()
    sessions = ChatSessions()

    if not no_batching:
        gen_queue.start(generator=runtime.run_chat)

    ChatHandler.web_root = str(resolved_root)
    server = ThreadingHTTPServer((host, port), ChatHandler)
    server.runtime = runtime  # type: ignore[attr-defined]
    server.gen_queue = gen_queue  # type: ignore[attr-defined]
    server.sessions = sessions  # type: ignore[attr-defined]
    server.control_port = None  # type: ignore[attr-defined]  # set by main() if an embedded control server starts
    return server


def shutdown_server(server: ThreadingHTTPServer):
    gen_queue = getattr(server, "gen_queue", None)
    if gen_queue is not None:
        gen_queue.stop()
    server.server_close()


def find_free_port(preferred: int) -> int:
    """Use the preferred port if free, otherwise let the OS pick one -- so a
    leftover process (or a second launch) doesn't just crash on startup with
    an address-in-use error. Shared by main() below and src/desktop_app.py."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")


def start_embedded_control_server(*, host: str, preferred_port: int, web_root: str):
    """Boot the control API (training/tasks/network/etc.) as a background
    thread alongside the chat server, so a plain `serve-web` browser tab gets
    the same Control room capability the desktop app has always had, instead
    of that panel being gated behind desktop_app.py specifically. Lazy import
    keeps `import src.serve_web` itself cheap for callers/tests that only
    need the chat server (serve_control.py pulls in federated/swarm/task
    machinery serve_web has no other reason to load)."""
    from src.serve_control import create_server as create_control_server

    control_port = find_free_port(preferred_port)
    control_server = create_control_server(host=host, port=control_port, web_root=web_root)
    control_thread = threading.Thread(target=control_server.serve_forever, daemon=True)
    control_thread.start()
    return control_server, control_port


def main():
    parser = argparse.ArgumentParser(description="Serve Ickle chat web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--web-root", default="web")
    parser.add_argument("--no-batching", action="store_true", help="Disable request batching")
    parser.add_argument("--control-port", type=int, default=8788)
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Chat only -- don't also start the control API (Training/Activity/Network/Sharing/etc. in the Control room).",
    )
    args = parser.parse_args()

    server = create_server(args.host, args.port, args.web_root, args.no_batching)

    control_server = None
    if not args.no_control:
        control_server, control_port = start_embedded_control_server(
            host=args.host, preferred_port=args.control_port, web_root=args.web_root
        )
        server.control_port = control_port  # type: ignore[attr-defined]
        print(f"Ickle control API: http://{args.host}:{control_port} (Control room: Training/Activity/Network/Sharing/etc.)")

    print(f"Ickle chat: http://{args.host}:{args.port}")
    print("API: /api/status, /api/chat, /api/chat/stream, /api/models, /api/flags, /api/control-port, /api/feedback, /api/epistemics/reviews")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping chat server...")
    finally:
        shutdown_server(server)
        if control_server is not None:
            from src.serve_control import shutdown_server as shutdown_control_server

            shutdown_control_server(control_server)


if __name__ == "__main__":
    main()
