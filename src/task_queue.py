from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


TaskRunner = Callable[[str, dict[str, Any], Callable[[str], None]], dict[str, Any]]
WORKER_LEASE_SECONDS = 900


class TaskCancelledError(RuntimeError):
    pass


def _is_fatal_exit_code(returncode: int) -> bool:
    if returncode >= 3221225472:
        return True
    if returncode in {134, 136, 139}:
        return True
    return False


def _is_fatal_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    patterns = [
        "heap_corruption",
        "access_violation",
        "illegal_instruction",
        "stack_buffer_overrun",
        "dll_not_found",
        "dll_init_failed",
        "forrtl: error",
        "status_heap_corruption",
        "status_access_violation",
        "status_illegal_instruction",
        "memory corruption",
        "segfault at",
        "sigsegv",
        "sigabrt",
        "sigill",
        "sigfpe",
        # Deterministic configuration mistakes (wrong/missing dataset field,
        # bad dataset id, missing config/subset) will fail identically on
        # every retry -- e.g. streaming Anthropic/hh-rlhf with the default
        # --stream-field "text" always yields 0 chars, since that dataset's
        # real fields are "chosen"/"rejected". Retrying these just burns two
        # attempts' worth of time before ever showing the user the message
        # that already correctly explains the fix.
        "produced too little text",
        "isn't a valid hugging face dataset id",
        "requires a config/subset name",
    ]
    for p in patterns:
        if p in msg:
            return True
    m = re.search(r"exit\s+(\d+)", msg)
    if m:
        code = int(m.group(1))
        if _is_fatal_exit_code(code):
            return True
    return False


TRAINING_TASK_TYPES = {
    "learn_wikipedia_topic",
    "learn_web_topic",
    "build_clean_corpus",
    "train_model",
    "continual_guard_step",
    "evaluate_model",
}


@dataclass
class TaskItem:
    task_id: str
    task_type: str
    payload: dict[str, Any]
    status: str
    created_at_utc: str
    updated_at_utc: str
    depends_on: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    progress: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    attempts: int = 0
    max_attempts: int = 1
    last_error_at_utc: str | None = None
    next_run_at_utc: str | None = None
    idempotency_key: str = ""


class TaskQueue:
    def __init__(
        self,
        path: str = "data/task_queue.json",
        *,
        max_parallel_tasks: int = 1,
        max_running_resource_percent: int = 100,
        recover_running_tasks: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._worker_stop = threading.Event()
        self._worker_threads: list[threading.Thread] = []
        self._worker_id = f"w_{uuid.uuid4().hex[:12]}"
        self._max_parallel_tasks = max(1, int(max_parallel_tasks))
        self._max_running_resource_percent = max(20, min(100, int(max_running_resource_percent)))
        if not self.path.exists():
            self._write_state({"tasks": []})
        if bool(recover_running_tasks):
            self._recover_running_tasks()

    def _normalize_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = payload if isinstance(payload, dict) else {}
        if "tasks" not in out or not isinstance(out["tasks"], list):
            out["tasks"] = []
        for task in out["tasks"]:
            task.setdefault("attempts", 0)
            task.setdefault("max_attempts", 1)
            task.setdefault("last_error_at_utc", None)
            task.setdefault("next_run_at_utc", None)
            task.setdefault("idempotency_key", "")
            task.setdefault("worker_id", "")
            task.setdefault("lease_expires_at_utc", None)
        return out

    def _archive_corrupt_state(self, raw_content: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = self.path.parent / f"{self.path.stem}.corrupt.{stamp}.{uuid.uuid4().hex[:6]}.json"
        try:
            archive.write_text(raw_content, encoding="utf-8")
        except Exception:  # noqa: BLE001
            return

    def _read_state(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            payload = {"tasks": []}
            self._write_state(payload)
            return payload

        if not raw.strip():
            payload = {"tasks": []}
            self._write_state(payload)
            return payload

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._archive_corrupt_state(raw)
            payload = {"tasks": []}
            self._write_state(payload)
            return payload

        return self._normalize_state(payload)

    def _write_state(self, payload: dict[str, Any]):
        normalized = self._normalize_state(payload)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.path)

    def _save_tasks(self, tasks: list[dict[str, Any]]):
        self._write_state({"tasks": tasks})

    def list_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            tasks = self._read_state()["tasks"]
            return list(reversed(tasks[-max(1, limit) :]))

    def add_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        depends_on: list[str] | None = None,
        *,
        idempotency_key: str = "",
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        with self._lock:
            normalized_key = str(idempotency_key or "").strip()
            state = self._read_state()
            if normalized_key:
                for existing in state["tasks"]:
                    if str(existing.get("idempotency_key", "")).strip() != normalized_key:
                        continue
                    if existing.get("status") in {"queued", "running"}:
                        return dict(existing)

            task = TaskItem(
                task_id=f"t_{uuid.uuid4().hex[:12]}",
                task_type=task_type,
                payload=payload,
                status="queued",
                created_at_utc=_utc_now(),
                updated_at_utc=_utc_now(),
                depends_on=list(depends_on or []),
                idempotency_key=normalized_key,
                max_attempts=max(1, int(max_attempts)),
            )
            state["tasks"].append(task.__dict__)
            self._save_tasks(state["tasks"])
            return task.__dict__

    def _set_task_fields(self, task_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        state = self._read_state()
        for task in state["tasks"]:
            if task["task_id"] == task_id:
                task.update(updates)
                task["updated_at_utc"] = _utc_now()
                self._save_tasks(state["tasks"])
                return task
        return None

    def _get_task(self, task_id: str) -> dict[str, Any] | None:
        state = self._read_state()
        for task in state["tasks"]:
            if task["task_id"] == task_id:
                return dict(task)
        return None

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            current = self._get_task(task_id)
            if not current:
                return False
            status = current.get("status")
            if status == "queued":
                task = self._set_task_fields(
                    task_id,
                    {"status": "cancelled", "progress": "cancelled by user", "next_run_at_utc": None},
                )
                return bool(task)
            if status == "running":
                task = self._set_task_fields(
                    task_id,
                    {"cancel_requested": True, "progress": "cancellation requested"},
                )
                # cancel_requested alone does nothing for training-shaped
                # tasks (train_model, and anything built on top of it --
                # train_from_teacher, lora_train, continual_guard_step):
                # their actual training subprocess only checks a separate,
                # file-based stop-request signal (src/training_control.py),
                # which nothing was writing to. Without this, "cancel" just
                # relabeled the task in the UI while the real subprocess (and
                # all the CPU/battery it uses) kept running to completion
                # unnoticed -- confirmed live when the generic /api/tasks/
                # <id>/cancel endpoint left a training process running after
                # the task list already showed it as cancelled.
                payload = current.get("payload") if isinstance(current.get("payload"), dict) else {}
                out_model = str(payload.get("out_model", "")).strip()
                if out_model:
                    from src.training_control import write_training_stop_request

                    # Must match run_train_model_task's own default exactly
                    # (src/task_actions.py) -- it computes checkpoint_path as
                    # payload["checkpoint_path"] or f"{out_model}.checkpoint.pt"
                    # when building the stop-request file's key, so passing a
                    # bare "" here when the payload has no explicit
                    # checkpoint_path would write to the wrong file (keyed by
                    # out_model alone) and the running subprocess would never
                    # see it.
                    checkpoint_path = str(payload.get("checkpoint_path") or f"{out_model}.checkpoint.pt").strip()
                    write_training_stop_request(
                        out_model=out_model,
                        checkpoint_path=checkpoint_path,
                    )
                return bool(task)
            return status in {"cancelled", "completed", "failed"}

    def _task_resource_cost(self, task: dict[str, Any]) -> int:
        kind = str(task.get("task_type", "")).strip().lower()
        payload = task.get("payload", {})
        budget = payload.get("resource_budget", {}) if isinstance(payload, dict) else {}
        if isinstance(budget, dict):
            values = []
            for key in ("cpu_percent", "ram_percent", "gpu_percent"):
                try:
                    value = int(float(budget.get(key, 0)))
                except Exception:  # noqa: BLE001
                    value = 0
                if value > 0:
                    values.append(value)
            if values:
                return max(5, min(100, max(values)))

        if kind in {"continual_guard_step", "train_model", "build_clean_corpus"}:
            return 70
        if kind in {"evaluate_model"}:
            return 45
        if kind in {"learn_web_topic", "learn_wikipedia_topic"}:
            return 20
        return 15

    def _next_queued_task(self) -> dict[str, Any] | None:
        state = self._read_state()
        status_by_id = {str(t.get("task_id", "")): str(t.get("status", "")) for t in state["tasks"]}
        now = datetime.now(timezone.utc)
        running_tasks = [t for t in state["tasks"] if str(t.get("status", "")) == "running"]
        if len(running_tasks) >= self._max_parallel_tasks:
            return None
        running_cost = sum(self._task_resource_cost(t) for t in running_tasks)

        for task in state["tasks"]:
            if task.get("status") == "queued":
                next_run = _parse_utc(task.get("next_run_at_utc"))
                if next_run and next_run > now:
                    continue
                deps = [str(dep) for dep in task.get("depends_on", []) if dep]
                if deps:
                    failed_deps = [dep for dep in deps if status_by_id.get(dep) in {"failed", "cancelled"}]
                    if failed_deps:
                        task["status"] = "cancelled"
                        task["error"] = f"Dependency failed/cancelled: {', '.join(failed_deps)}"
                        task["progress"] = "cancelled due to dependency failure"
                        task["updated_at_utc"] = _utc_now()
                        self._save_tasks(state["tasks"])
                        status_by_id[str(task.get("task_id", ""))] = "cancelled"
                        continue
                    waiting = [dep for dep in deps if status_by_id.get(dep) != "completed"]
                    if waiting:
                        continue
                candidate_cost = self._task_resource_cost(task)
                if running_tasks and (running_cost + candidate_cost > self._max_running_resource_percent):
                    continue
                task["status"] = "running"
                task["progress"] = task.get("progress") or "running"
                task["next_run_at_utc"] = None
                task["worker_id"] = self._worker_id
                task["lease_expires_at_utc"] = (
                    now + timedelta(seconds=WORKER_LEASE_SECONDS)
                ).isoformat()
                task["updated_at_utc"] = _utc_now()
                self._save_tasks(state["tasks"])
                return task
        return None

    def _recover_running_tasks(self):
        with self._lock:
            state = self._read_state()
            changed = False
            now = datetime.now(timezone.utc)
            for task in state["tasks"]:
                if task.get("status") != "running":
                    continue
                lease_expires = _parse_utc(task.get("lease_expires_at_utc"))
                if lease_expires is not None and lease_expires > now:
                    continue
                task["status"] = "failed"
                task["progress"] = "interrupted: worker lease expired"
                task["error"] = (
                    "The worker stopped before completing this task. "
                    "Retry it explicitly; Ickle will not restart abandoned compute automatically."
                )
                task["worker_id"] = ""
                task["lease_expires_at_utc"] = None
                task["updated_at_utc"] = _utc_now()
                changed = True
            if changed:
                self._save_tasks(state["tasks"])

    def start_worker(
        self,
        runner: TaskRunner,
        poll_seconds: float = 1.0,
        *,
        max_parallel_tasks: int | None = None,
        max_running_resource_percent: int | None = None,
    ):
        with self._lock:
            desired_parallel = self._max_parallel_tasks if max_parallel_tasks is None else max(1, int(max_parallel_tasks))
            desired_resource = (
                self._max_running_resource_percent
                if max_running_resource_percent is None
                else max(20, min(100, int(max_running_resource_percent)))
            )
            alive_count = sum(1 for t in self._worker_threads if t.is_alive())
            if (
                alive_count > 0
                and desired_parallel == self._max_parallel_tasks
                and desired_resource == self._max_running_resource_percent
            ):
                return
            if alive_count > 0:
                self.stop_worker()

            self._max_parallel_tasks = desired_parallel
            self._max_running_resource_percent = desired_resource
            self._worker_stop.clear()

            def _loop():
                while not self._worker_stop.is_set():
                    task = None
                    with self._lock:
                        task = self._next_queued_task()
                    if not task:
                        time.sleep(max(0.1, poll_seconds))
                        continue

                    task_id = str(task["task_id"])

                    def progress_cb(message: str):
                        with self._lock:
                            current = self._get_task(task_id)
                            if not current:
                                raise TaskCancelledError("Task disappeared while running.")
                            if current.get("status") == "cancelled" or bool(current.get("cancel_requested", False)):
                                raise TaskCancelledError("Cancelled by user.")
                            updates: dict[str, Any] = {
                                "progress": message,
                                "worker_id": self._worker_id,
                                "lease_expires_at_utc": (
                                    datetime.now(timezone.utc)
                                    + timedelta(seconds=WORKER_LEASE_SECONDS)
                                ).isoformat(),
                            }
                            if current.get("status") != "running":
                                updates["status"] = "running"
                            self._set_task_fields(task_id, updates)

                    try:
                        result = runner(task["task_type"], task.get("payload", {}), progress_cb)
                        with self._lock:
                            self._set_task_fields(
                                task_id,
                                {
                                    "status": "completed",
                                    "result": result,
                                    "progress": "completed",
                                    "cancel_requested": False,
                                    "error": "",
                                    "next_run_at_utc": None,
                                    "worker_id": "",
                                    "lease_expires_at_utc": None,
                                },
                            )
                    except TaskCancelledError:
                        with self._lock:
                            self._set_task_fields(
                                task_id,
                                {
                                    "status": "cancelled",
                                    "progress": "cancelled by user",
                                    "cancel_requested": False,
                                    "error": "",
                                    "next_run_at_utc": None,
                                    "worker_id": "",
                                    "lease_expires_at_utc": None,
                                },
                            )
                    except Exception as exc:  # noqa: BLE001
                        with self._lock:
                            current = self._get_task(task_id) or {}
                            attempts = int(current.get("attempts", 0)) + 1
                            max_attempts = max(1, int(current.get("max_attempts", 1)))
                            is_fatal = _is_fatal_error(exc)
                            if is_fatal:
                                attempts = max_attempts
                            if attempts < max_attempts:
                                backoff_seconds = min(300, 2**attempts)
                                next_run = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)
                                self._set_task_fields(
                                    task_id,
                                    {
                                        "status": "queued",
                                        "attempts": attempts,
                                        "error": str(exc),
                                        "last_error_at_utc": _utc_now(),
                                        "cancel_requested": False,
                                        "progress": (
                                            f"retry {attempts}/{max_attempts} scheduled in "
                                            f"{backoff_seconds}s"
                                        ),
                                        "next_run_at_utc": next_run.isoformat(),
                                        "worker_id": "",
                                        "lease_expires_at_utc": None,
                                    },
                                )
                            else:
                                self._set_task_fields(
                                    task_id,
                                    {
                                        "status": "failed",
                                        "attempts": attempts,
                                        "error": str(exc),
                                        "last_error_at_utc": _utc_now(),
                                        "cancel_requested": False,
                                        "progress": "failed (fatal)" if is_fatal else "failed",
                                        "next_run_at_utc": None,
                                        "worker_id": "",
                                        "lease_expires_at_utc": None,
                                    },
                                )

            self._worker_threads = []
            for idx in range(self._max_parallel_tasks):
                thread = threading.Thread(
                    target=_loop,
                    name=f"task-queue-worker-{idx + 1}",
                    daemon=True,
                )
                thread.start()
                self._worker_threads.append(thread)

    def stop_worker(self):
        self._worker_stop.set()
        for thread in list(self._worker_threads):
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._worker_threads = []
