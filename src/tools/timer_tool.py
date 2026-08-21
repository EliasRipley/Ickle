from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Timer:
    name: str
    duration_seconds: float
    remaining_seconds: float
    created_at: float
    status: str = "running"  # running, paused, completed, cancelled
    callback_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "remaining_seconds": self.remaining_seconds,
            "created_at": self.created_at,
            "status": self.status,
            "callback_message": self.callback_message,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Timer":
        return Timer(
            name=d.get("name", ""),
            duration_seconds=d.get("duration_seconds", 0),
            remaining_seconds=d.get("remaining_seconds", 0),
            created_at=d.get("created_at", time.time()),
            status=d.get("status", "running"),
            callback_message=d.get("callback_message", ""),
        )


class TimerManager:
    _instance: TimerManager | None = None
    _lock = threading.Lock()

    def __init__(self, state_dir: str | None = None):
        self.timers: dict[str, Timer] = {}
        self._tick_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_dir = Path(state_dir) if state_dir else Path("data")
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._state_dir / "timers.json"
        self._tick_interval = 0.5
        self._load_state()
        self._start_tick_thread()

    @classmethod
    def get_instance(cls, state_dir: str | None = None) -> "TimerManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(state_dir=state_dir)
        return cls._instance

    def _load_state(self):
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for name, td in raw.items():
                    self.timers[name] = Timer.from_dict(td)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        payload = {name: t.to_dict() for name, t in self.timers.items()}
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _start_tick_thread(self):
        if self._tick_thread is not None:
            return
        self._stop_event.clear()
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

    def _tick_loop(self):
        last_save = time.time()
        while not self._stop_event.is_set():
            self._stop_event.wait(self._tick_interval)
            now = time.time()
            changed = False
            with self._lock:
                for timer in list(self.timers.values()):
                    if timer.status == "running":
                        timer.remaining_seconds = max(0, timer.remaining_seconds - self._tick_interval)
                        if timer.remaining_seconds <= 0:
                            timer.status = "completed"
                            timer.remaining_seconds = 0
                            timer.callback_message = f"Timer '{timer.name}' completed."
                        changed = True
            if changed and now - last_save >= 2.0:
                self._save_state()
                last_save = now

    def set_timer(self, name: str, duration_seconds: float, callback_message: str = "") -> Timer:
        with self._lock:
            name = name.strip()
            if not name:
                name = f"timer_{len(self.timers) + 1}"
            timer = Timer(
                name=name,
                duration_seconds=duration_seconds,
                remaining_seconds=duration_seconds,
                created_at=time.time(),
                status="running",
                callback_message=callback_message,
            )
            self.timers[name] = timer
            self._save_state()
            return timer

    def cancel_timer(self, name: str) -> Timer | None:
        with self._lock:
            timer = self.timers.get(name)
            if timer and timer.status in ("running", "paused"):
                timer.status = "cancelled"
                timer.remaining_seconds = 0
                self._save_state()
                return timer
            return None

    def pause_timer(self, name: str) -> Timer | None:
        with self._lock:
            timer = self.timers.get(name)
            if timer and timer.status == "running":
                timer.status = "paused"
                self._save_state()
                return timer
            return None

    def resume_timer(self, name: str) -> Timer | None:
        with self._lock:
            timer = self.timers.get(name)
            if timer and timer.status == "paused":
                timer.status = "running"
                self._save_state()
                return timer
            return None

    def get_timer(self, name: str) -> Timer | None:
        with self._lock:
            return self.timers.get(name)

    def list_timers(self) -> list[Timer]:
        with self._lock:
            return list(self.timers.values())

    def check_completed(self) -> list[Timer]:
        completed = []
        with self._lock:
            for name, timer in list(self.timers.items()):
                if timer.status == "completed":
                    if timer.callback_message:
                        completed.append(timer)
                    del self.timers[name]
            if completed:
                self._save_state()
        return completed

    def shutdown(self):
        self._stop_event.set()
        if self._tick_thread:
            self._tick_thread.join(timeout=2.0)
        self._save_state()


def format_timer_status(timer: Timer) -> str:
    remaining = timer.remaining_seconds
    if timer.status == "completed":
        return f"Timer '{timer.name}': COMPLETED"
    elif timer.status == "cancelled":
        return f"Timer '{timer.name}': CANCELLED"
    elif timer.status == "paused":
        mins, secs = divmod(int(remaining), 60)
        return f"Timer '{timer.name}': PAUSED at {mins}m {secs}s remaining"
    elif timer.status == "running":
        mins, secs = divmod(int(remaining), 60)
        return f"Timer '{timer.name}': {mins}m {secs}s remaining"
    return f"Timer '{timer.name}': {timer.status}"


def parse_duration(text: str) -> float | None:
    text = text.strip().lower()
    total = 0.0
    parts = text.replace(",", " ").split()
    i = 0
    while i < len(parts):
        part = parts[i]
        try:
            num = float(part)
        except ValueError:
            i += 1
            continue
        i += 1
        if i < len(parts):
            unit = parts[i].rstrip("s.,;")
            i += 1
        else:
            unit = "s"
        if unit in ("h", "hr", "hour", "hours"):
            total += num * 3600
        elif unit in ("m", "min", "minute", "minutes"):
            total += num * 60
        else:
            total += num
    return total if total > 0 else None
