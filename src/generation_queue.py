"""Generation queue with request batching for concurrent chat throughput."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

_MAX_BATCH_SIZE = 8
_MAX_WAIT_MS = 50
_MAX_QUEUE_DEPTH = 64


class _PendingRequest:
    __slots__ = ("payload", "event", "result", "submitted_at")

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.event = threading.Event()
        self.result: dict[str, Any] | None = None
        self.submitted_at = time.monotonic()


class GenerationQueue:
    """Batches concurrent /api/chat requests to improve GPU/CPU utilisation.

    Instead of running one forward pass per request, the queue collects
    requests for up to `max_wait_ms` or `max_batch_size` items, then
    executes them sequentially without interleaving â€” better L2/L3 cache
    locality and less thread-hopping on CPU inference.
    """

    def __init__(
        self,
        *,
        max_batch_size: int = _MAX_BATCH_SIZE,
        max_wait_ms: int = _MAX_WAIT_MS,
        max_queue_depth: int = _MAX_QUEUE_DEPTH,
    ):
        self._max_batch = max(1, min(16, max_batch_size))
        self._max_wait = max(5, min(500, max_wait_ms)) / 1000.0
        self._max_depth = max(1, max_queue_depth)
        self._lock = threading.Lock()
        self._pending: deque[_PendingRequest] = deque()
        self._drain_event = threading.Event()
        self._worker_running = False
        self._drain_thread: threading.Thread | None = None

    def _drain_loop(self, generator: Callable[[dict[str, Any]], dict[str, Any]]):
        while self._worker_running:
            batch: list[_PendingRequest] = []
            deadline = time.monotonic() + self._max_wait
            while time.monotonic() < deadline:
                with self._lock:
                    while self._pending and len(batch) < self._max_batch:
                        batch.append(self._pending.popleft())
                if len(batch) >= self._max_batch:
                    break
                self._drain_event.wait(timeout=min(0.005, self._max_wait / 4))
                self._drain_event.clear()

            with self._lock:
                while self._pending and len(batch) < self._max_batch:
                    batch.append(self._pending.popleft())

            if not batch:
                continue

            for req in batch:
                try:
                    req.result = generator(req.payload)
                except Exception as exc:
                    req.result = {"response": "", "error": str(exc)}
                req.event.set()

    def start(self, generator: Callable[[dict[str, Any]], dict[str, Any]]):
        with self._lock:
            if self._worker_running:
                return
            self._worker_running = True
            self._drain_thread = threading.Thread(
                target=self._drain_loop,
                args=(generator,),
                daemon=True,
                name="gen-queue-drain",
            )
            self._drain_thread.start()

    def stop(self):
        with self._lock:
            self._worker_running = False
        self._drain_event.set()
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=5.0)

    def enqueue(self, payload: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
        with self._lock:
            if len(self._pending) >= self._max_depth:
                return {"response": "", "error": "Queue full, try again shortly."}

            req = _PendingRequest(payload)
            self._pending.append(req)

        self._drain_event.set()

        if not req.event.wait(timeout=timeout_s):
            return {"response": "", "error": "Request timed out."}

        return req.result or {"response": "", "error": "No result produced."}
