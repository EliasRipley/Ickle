import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from src.runtime_flags import RuntimeFlagsStore
from src.serve_control import ControlRuntime
from src.task_queue import TaskQueue


class ControlRuntimeWorkerStartupTests(unittest.TestCase):
    """Regression coverage for a real bug: the background task worker only
    ever started reactively, inside create_task()/update_flags() calling
    _sync_worker() -- never automatically on server startup. A server
    restarted with tasks already sitting in the persisted queue left them
    stuck at "queued" forever until something incidental (a new task, a
    flags change) happened to trigger _sync_worker(). Confirmed live: a
    freshly restarted serve-web instance did not touch an already-queued
    training task until an unrelated flags update kicked the worker."""

    def test_init_starts_worker_without_any_external_trigger(self):
        with tempfile.TemporaryDirectory() as td:
            tasks_path = str(Path(td) / "task_queue.json")
            flags_path = str(Path(td) / "runtime_flags.json")

            queue = TaskQueue(path=tasks_path)
            queue.add_task(task_type="noop", payload={})

            with mock.patch("src.serve_control.TaskQueue", return_value=queue), \
                 mock.patch("src.serve_control.RuntimeFlagsStore", return_value=RuntimeFlagsStore(flags_path)), \
                 mock.patch.object(ControlRuntime, "_init_swarm", lambda self: None):
                runtime = ControlRuntime(training_root=td)

            # The worker thread(s) must already be alive right after
            # __init__ returns -- no create_task/update_flags call in between.
            alive = [t for t in queue._worker_threads if t.is_alive()]
            self.assertTrue(alive, "worker thread did not start during __init__")

            deadline = time.monotonic() + 5.0
            picked_up = False
            while time.monotonic() < deadline:
                task = queue._get_task(list(queue._read_state()["tasks"])[0]["task_id"])
                if task and task["status"] != "queued":
                    picked_up = True
                    break
                time.sleep(0.1)
            self.assertTrue(picked_up, "pre-existing queued task was never picked up after startup")

            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
