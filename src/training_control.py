from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str, *, max_len: int = 140) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    if not text:
        return "default"
    return text[:max_len]


def get_training_stop_request_path(*, out_model: str, checkpoint_path: str = "") -> Path:
    key = str(checkpoint_path or out_model or "default").strip()
    slug = _safe_slug(key)
    return Path("data/runtime/training_stop") / f"{slug}.json"


def write_training_stop_request(
    *,
    out_model: str,
    checkpoint_path: str = "",
    request: dict[str, Any] | None = None,
) -> Path:
    path = get_training_stop_request_path(out_model=out_model, checkpoint_path=checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_at_utc": _utc_now(),
        "request_type": "graceful_stop",
    }
    if isinstance(request, dict):
        payload.update(request)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_training_stop_request(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def clear_training_stop_request(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _parse_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            return False
    except (psutil.Error, OSError, ValueError):
        return False


def inspect_training_status(
    path: str | Path,
    *,
    stale_after_seconds: int = 180,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read a trainer heartbeat and distinguish live work from abandoned state."""
    status_path = Path(path)
    result: dict[str, Any] = {
        "path": str(status_path),
        "exists": status_path.exists(),
        "status": "unavailable",
        "reported_status": "",
        "is_active": False,
        "is_stale": False,
        "age_seconds": None,
        "process_alive": None,
        "stale_reason": "",
    }
    if not status_path.exists():
        return result
    try:
        loaded = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.update({"status": "invalid", "error": str(exc)})
        return result
    if not isinstance(loaded, dict):
        result.update({"status": "invalid", "error": "Status document must be an object"})
        return result

    result.update(loaded)
    reported = str(loaded.get("status", "unknown") or "unknown").strip().lower()
    result["reported_status"] = reported
    result["status"] = reported

    timestamp = _parse_utc(loaded.get("timestamp_utc"))
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age_seconds: float | None = None
    if timestamp is not None:
        age_seconds = max(0.0, (current.astimezone(timezone.utc) - timestamp).total_seconds())
        result["age_seconds"] = round(age_seconds, 1)

    pid = 0
    try:
        pid = int(loaded.get("pid", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    process_alive: bool | None = _pid_is_running(pid) if pid > 0 else None
    result["process_alive"] = process_alive

    if reported == "running":
        stale_reason = ""
        if timestamp is None:
            stale_reason = "missing heartbeat timestamp"
        elif age_seconds is not None and age_seconds > max(1, int(stale_after_seconds)):
            stale_reason = "heartbeat expired"
        elif process_alive is False:
            stale_reason = "trainer process is not running"
        if stale_reason:
            result.update(
                {
                    "status": "stale",
                    "is_active": False,
                    "is_stale": True,
                    "stale_reason": stale_reason,
                }
            )
        else:
            result["is_active"] = True
    return result
