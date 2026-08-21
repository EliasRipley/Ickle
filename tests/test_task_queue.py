import time
import threading
import unittest
import uuid
from pathlib import Path

from src.task_queue import TaskQueue, _is_fatal_error
from src.training_control import get_training_stop_request_path


class FatalErrorClassificationTests(unittest.TestCase):
    def test_wrong_stream_field_is_fatal(self):
        exc = ValueError("Streamed dataset Anthropic/hh-rlhf produced too little text (0 chars). Check --stream-filter and --stream-field.")
        self.assertTrue(_is_fatal_error(exc))

    def test_bad_dataset_id_is_fatal(self):
        exc = ValueError("'wikitext' isn't a valid Hugging Face dataset id -- it needs an organization/user prefix.")
        self.assertTrue(_is_fatal_error(exc))

    def test_missing_config_is_fatal(self):
        exc = ValueError("Dataset 'Salesforce/wikitext' requires a config/subset name (e.g. --stream-config wikitext-2-raw-v1).")
        self.assertTrue(_is_fatal_error(exc))

    def test_transient_network_error_is_not_fatal(self):
        exc = ConnectionError("Connection reset by peer while downloading shard")
        self.assertFalse(_is_fatal_error(exc))


class TaskQueueTests(unittest.TestCase):
    def _path(self) -> str:
        root = Path("data/test_task_queue")
        root.mkdir(parents=True, exist_ok=True)
        return str(root / f"queue_{uuid.uuid4().hex}.json")

    def test_add_and_list_tasks(self):
        queue = TaskQueue(path=self._path())
        queue.add_task("learn_wikipedia_topic", {"topic": "test"})
        tasks = queue.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_type"], "learn_wikipedia_topic")

    def test_worker_completes_task(self):
        queue = TaskQueue(path=self._path())
        queue.add_task("mock_task", {"a": 1})

        def runner(task_type, payload, progress):
            progress("working")
            return {"task_type": task_type, "payload": payload}

        queue.start_worker(runner, poll_seconds=0.05)
        deadline = time.time() + 3.0
        status = ""
        while time.time() < deadline:
            tasks = queue.list_tasks()
            status = tasks[0]["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        queue.stop_worker()

        self.assertEqual(status, "completed")
        tasks = queue.list_tasks()
        self.assertEqual(tasks[0]["result"]["task_type"], "mock_task")

    def test_cancel_running_task(self):
        queue = TaskQueue(path=self._path())
        task = queue.add_task("mock_task", {"a": 1})

        def runner(task_type, payload, progress):
            for _ in range(200):
                progress("working")
                time.sleep(0.01)
            return {"task_type": task_type, "payload": payload}

        queue.start_worker(runner, poll_seconds=0.01)
        deadline = time.time() + 4.0
        seen_running = False
        while time.time() < deadline:
            current = queue.list_tasks()[0]
            if current["status"] == "running":
                seen_running = True
                break
            time.sleep(0.02)
        self.assertTrue(seen_running)

        queue.cancel_task(task["task_id"])

        deadline = time.time() + 4.0
        status = ""
        while time.time() < deadline:
            current = queue.list_tasks()[0]
            status = current["status"]
            if status == "cancelled":
                break
            time.sleep(0.02)
        queue.stop_worker()
        self.assertEqual(status, "cancelled")

    def test_cancel_running_train_model_task_writes_stop_request(self):
        """Regression test for a real bug: cancelling a running train_model
        task only ever set cancel_requested on the task record, which
        run_train_model_task (src/task_actions.py) never checks -- it only
        watches a separate file-based stop-request signal
        (src/training_control.py) that nothing was writing to. Confirmed
        live: the task list showed "cancelled" while the actual training
        subprocess kept running (and burning CPU) until manually killed."""
        out_model = f"models/candidates/test_cancel_{uuid.uuid4().hex}.pt"
        stop_path = get_training_stop_request_path(
            out_model=out_model, checkpoint_path=f"{out_model}.checkpoint.pt"
        )
        self.addCleanup(lambda: stop_path.unlink(missing_ok=True))
        self.assertFalse(stop_path.exists())

        queue = TaskQueue(path=self._path())
        task = queue.add_task("train_model", {"out_model": out_model, "steps": 100})

        def runner(task_type, payload, progress):
            for _ in range(200):
                progress("training")
                time.sleep(0.01)
            return {}

        queue.start_worker(runner, poll_seconds=0.01)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if queue.list_tasks()[0]["status"] == "running":
                break
            time.sleep(0.02)

        queue.cancel_task(task["task_id"])
        queue.stop_worker()

        # checkpoint_path wasn't given explicitly -- must match
        # run_train_model_task's own default (out_model + ".checkpoint.pt"),
        # not a bare out_model-only key, or the running subprocess would
        # never see this file.
        self.assertTrue(stop_path.exists())

    def test_cancel_running_train_model_task_respects_explicit_checkpoint_path(self):
        out_model = f"models/candidates/test_cancel2_{uuid.uuid4().hex}.pt"
        checkpoint_path = f"models/candidates/test_cancel2_custom_{uuid.uuid4().hex}.pt"
        stop_path = get_training_stop_request_path(out_model=out_model, checkpoint_path=checkpoint_path)
        self.addCleanup(lambda: stop_path.unlink(missing_ok=True))

        queue = TaskQueue(path=self._path())
        task = queue.add_task("train_model", {"out_model": out_model, "checkpoint_path": checkpoint_path})

        def runner(task_type, payload, progress):
            for _ in range(200):
                progress("training")
                time.sleep(0.01)
            return {}

        queue.start_worker(runner, poll_seconds=0.01)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if queue.list_tasks()[0]["status"] == "running":
                break
            time.sleep(0.02)

        queue.cancel_task(task["task_id"])
        queue.stop_worker()

        self.assertTrue(stop_path.exists())

    def test_cancel_running_non_training_task_does_not_write_stop_request(self):
        queue = TaskQueue(path=self._path())
        task = queue.add_task("learn_wikipedia_topic", {"topic": "x"})

        def runner(task_type, payload, progress):
            for _ in range(200):
                progress("working")
                time.sleep(0.01)
            return {}

        queue.start_worker(runner, poll_seconds=0.01)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if queue.list_tasks()[0]["status"] == "running":
                break
            time.sleep(0.02)

        # No out_model in payload -- nothing to key a stop-request file by,
        # and nothing should be written.
        from unittest import mock

        with mock.patch("src.training_control.write_training_stop_request") as mock_write:
            queue.cancel_task(task["task_id"])
        queue.stop_worker()
        mock_write.assert_not_called()

    def test_dependency_failure_cancels_followup(self):
        queue = TaskQueue(path=self._path())
        first = queue.add_task("first", {"a": 1})
        queue.add_task("second", {"a": 1}, depends_on=[first["task_id"]])

        def runner(task_type, payload, progress):
            if task_type == "first":
                raise RuntimeError("boom")
            return {"ok": True}

        queue.start_worker(runner, poll_seconds=0.01)
        deadline = time.time() + 4.0
        while time.time() < deadline:
            tasks = queue.list_tasks(limit=10)
            statuses = {t["task_type"]: t["status"] for t in tasks}
            if statuses.get("first") in {"failed", "completed", "cancelled"} and statuses.get("second") in {
                "failed",
                "completed",
                "cancelled",
            }:
                break
            time.sleep(0.02)
        queue.stop_worker()

        tasks = {t["task_type"]: t for t in queue.list_tasks(limit=10)}
        self.assertEqual(tasks["first"]["status"], "failed")
        self.assertEqual(tasks["second"]["status"], "cancelled")

    def test_idempotency_key_reuses_active_task(self):
        queue = TaskQueue(path=self._path())
        first = queue.add_task("learn_wikipedia_topic", {"topic": "x"}, idempotency_key="topic:x")
        second = queue.add_task("learn_wikipedia_topic", {"topic": "x"}, idempotency_key="topic:x")
        self.assertEqual(first["task_id"], second["task_id"])

    def test_retry_then_complete(self):
        queue = TaskQueue(path=self._path())
        queue.add_task("flaky", {}, max_attempts=3)
        calls = {"count": 0}

        def runner(task_type, payload, progress):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient")
            progress("ok")
            return {"ok": True}

        queue.start_worker(runner, poll_seconds=0.01)
        deadline = time.time() + 8.0
        status = ""
        while time.time() < deadline:
            task = queue.list_tasks()[0]
            status = task["status"]
            if status == "completed":
                break
            time.sleep(0.05)
        queue.stop_worker()
        self.assertEqual(status, "completed")
        self.assertGreaterEqual(calls["count"], 2)

    def test_orphaned_running_task_is_not_restarted_automatically(self):
        path = self._path()
        queue = TaskQueue(path=path)
        task = queue.add_task("x", {})
        queue._set_task_fields(task["task_id"], {"status": "running"})
        recovered = TaskQueue(path=path)
        tasks = recovered.list_tasks()
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertIn("lease expired", tasks[0]["progress"])

    def test_live_worker_lease_is_not_recovered_by_an_observer(self):
        path = self._path()
        queue = TaskQueue(path=path)
        task = queue.add_task("x", {})
        queue._set_task_fields(
            task["task_id"],
            {
                "status": "running",
                "worker_id": "w_live",
                "lease_expires_at_utc": "2999-01-01T00:00:00+00:00",
            },
        )
        observer = TaskQueue(path=path)
        self.assertEqual(observer.list_tasks()[0]["status"], "running")

    def test_corrupt_state_auto_recovers(self):
        path = Path(self._path())
        path.write_text('{"tasks":[{"task_id":"broken","status":"queued"}', encoding="utf-8")

        queue = TaskQueue(path=str(path))
        tasks = queue.list_tasks(limit=5)
        self.assertEqual(tasks, [])

        backups = sorted(path.parent.glob(f"{path.stem}.corrupt.*.json"))
        self.assertTrue(backups)
        self.assertIn('"task_id":"broken"', backups[-1].read_text(encoding="utf-8"))

    def test_parallel_workers_run_when_budget_allows(self):
        queue = TaskQueue(path=self._path(), max_parallel_tasks=2, max_running_resource_percent=90)
        queue.add_task("train_model", {"resource_budget": {"cpu_percent": 40, "ram_percent": 40}})
        queue.add_task("train_model", {"resource_budget": {"cpu_percent": 40, "ram_percent": 40}})

        lock = threading.Lock()
        active = {"count": 0, "max": 0}

        def runner(task_type, payload, progress):
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.20)
            with lock:
                active["count"] -= 1
            return {"ok": True}

        queue.start_worker(runner, poll_seconds=0.01, max_parallel_tasks=2, max_running_resource_percent=90)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            tasks = queue.list_tasks(limit=10)
            if tasks and all(t["status"] == "completed" for t in tasks):
                break
            time.sleep(0.02)
        queue.stop_worker()
        self.assertGreaterEqual(active["max"], 2)

    def test_parallel_workers_block_when_budget_would_exceed_limit(self):
        queue = TaskQueue(path=self._path(), max_parallel_tasks=2, max_running_resource_percent=90)
        queue.add_task("train_model", {"resource_budget": {"cpu_percent": 70, "ram_percent": 70}})
        queue.add_task("train_model", {"resource_budget": {"cpu_percent": 70, "ram_percent": 70}})

        lock = threading.Lock()
        active = {"count": 0, "max": 0}

        def runner(task_type, payload, progress):
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            time.sleep(0.20)
            with lock:
                active["count"] -= 1
            return {"ok": True}

        queue.start_worker(runner, poll_seconds=0.01, max_parallel_tasks=2, max_running_resource_percent=90)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            tasks = queue.list_tasks(limit=10)
            if tasks and all(t["status"] == "completed" for t in tasks):
                break
            time.sleep(0.02)
        queue.stop_worker()
        self.assertEqual(active["max"], 1)


if __name__ == "__main__":
    unittest.main()
