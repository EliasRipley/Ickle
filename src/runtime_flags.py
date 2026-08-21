from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_FLAGS: dict[str, Any] = {
    "chat_enabled": True,
    "web_tools_enabled": True,
    "memory_enabled": True,
    "background_task_worker_enabled": True,
    # Off by default: this is the real switch for whether the local swarm
    # node (src/serve_control.py's ControlRuntime._init_swarm) binds
    # externally and joins known peers, vs. staying loopback-only/passive --
    # matching the UI's "nothing sent anywhere unless you turn it on" promise.
    "federated_enabled": False,
    "allow_auto_training_tasks": True,
    "parallel_training_enabled": True,
    "max_parallel_training_tasks": 2,
    "current_model": "",
}


class RuntimeFlagsStore:
    def __init__(self, path: str = "data/runtime_flags.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(DEFAULT_FLAGS)

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        out = dict(DEFAULT_FLAGS)
        if isinstance(payload, dict):
            out.update(payload)
        return out

    def _write(self, payload: dict[str, Any]):
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.path)

    def get_flags(self) -> dict[str, Any]:
        return self._read()

    def set_flag(self, key: str, value: Any):
        payload = self._read()
        payload[key] = value
        self._write(payload)

    def update_flags(self, updates: dict[str, Any]) -> dict[str, Any]:
        payload = self._read()
        for key, value in updates.items():
            if key in DEFAULT_FLAGS:
                payload[key] = value
        self._write(payload)
        return payload
