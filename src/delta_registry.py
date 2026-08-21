import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DeltaRegistry:
    def __init__(self, data_dir: str = "data/deltas"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._deltas: dict[str, dict[str, Any]] = {}
        self._load()

    def _registry_path(self) -> Path:
        return self.data_dir / "delta_registry.json"

    def _delta_dir(self, delta_id: str) -> Path:
        return self.data_dir / delta_id

    def _load(self):
        p = self._registry_path()
        if p.exists():
            try:
                self._deltas = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._deltas = {}

    def _save(self):
        self._registry_path().write_text(json.dumps(self._deltas, indent=2, ensure_ascii=False), encoding="utf-8")

    def register(self, delta_id: str, version: str = "1.0.0",
                 domain_description: str = "", description: str = "",
                 adapter_path: str = "", memory_entries: list[dict] | None = None,
                 rank: int = 8, alpha: int = 16,
                 activation_threshold: float = 0.6,
                 confidence: float = 1.0, provenance: str = "") -> dict[str, Any]:
        if delta_id in self._deltas:
            existing_version = self._deltas[delta_id].get("version", "1.0.0")
            rollback_dir = self._delta_dir(delta_id) / f"v{existing_version}"
            self._deltas[delta_id]["rollback_pointer"] = str(rollback_dir)

        self._deltas[delta_id] = {
            "delta_id": delta_id, "version": version,
            "domain_description": domain_description,
            "description": description,
            "adapter_path": adapter_path,
            "memory_entries": memory_entries or [],
            "rank": rank, "alpha": alpha,
            "activation_threshold": activation_threshold,
            "confidence": confidence, "provenance": provenance,
            "enabled": True,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return dict(self._deltas[delta_id])

    def get(self, delta_id: str) -> dict[str, Any] | None:
        entry = self._deltas.get(delta_id)
        return dict(entry) if entry else None

    def list_all(self) -> list[dict[str, Any]]:
        return [dict(d) for d in self._deltas.values()]

    def list_enabled(self) -> list[dict[str, Any]]:
        return [dict(d) for d in self._deltas.values() if d.get("enabled", True)]

    def disable(self, delta_id: str) -> bool:
        if delta_id not in self._deltas:
            return False
        self._deltas[delta_id]["enabled"] = False
        self._deltas[delta_id]["disabled_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def enable(self, delta_id: str) -> bool:
        if delta_id not in self._deltas:
            return False
        self._deltas[delta_id]["enabled"] = True
        self._deltas[delta_id].pop("disabled_at_utc", None)
        self._save()
        return True

    def rollback(self, delta_id: str) -> dict[str, Any] | None:
        entry = self._deltas.get(delta_id)
        if not entry:
            return None
        rollback = entry.get("rollback_pointer", "")
        if not rollback:
            return None
        rp = Path(rollback)
        if not rp.exists():
            return None
        try:
            self._deltas[delta_id] = json.loads((rp / "entry.json").read_text(encoding="utf-8"))
            self._deltas[delta_id]["enabled"] = True
            self._deltas[delta_id]["rolled_back_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return dict(self._deltas[delta_id])
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def remove(self, delta_id: str) -> bool:
        if delta_id not in self._deltas:
            return False
        self._deltas.pop(delta_id)
        dd = self._delta_dir(delta_id)
        if dd.exists():
            shutil.rmtree(str(dd), ignore_errors=True)
        self._save()
        return True

    def save_version(self, delta_id: str):
        entry = self._deltas.get(delta_id)
        if not entry:
            return
        dd = self._delta_dir(delta_id) / f"v{entry.get('version', '1.0.0')}"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / "entry.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_eval_results(self, delta_id: str, results: dict[str, Any]):
        if delta_id in self._deltas:
            self._deltas[delta_id]["evaluation_results"] = results
            self._save()

    def update_threshold(self, delta_id: str, threshold: float):
        if delta_id in self._deltas:
            self._deltas[delta_id]["activation_threshold"] = threshold
            self._save()

    def inspect(self, delta_id: str) -> str:
        entry = self.get(delta_id)
        if not entry:
            return f"Delta '{delta_id}' not found."
        lines = [
            f"Delta: {entry['delta_id']} v{entry['version']}",
            f"  Description: {entry.get('description', 'none')}",
            f"  Domain: {entry.get('domain_description', 'none')[:120]}",
            f"  Enabled: {entry.get('enabled', True)}",
            f"  Confidence: {entry.get('confidence', 1.0)}",
            f"  Threshold: {entry.get('activation_threshold', 0.6)}",
            f"  Provenance: {entry.get('provenance', 'unknown')}",
            f"  Adapter: {entry.get('adapter_path', 'none')} (r={entry.get('rank', 8)}, a={entry.get('alpha', 16)})",
            f"  Memory entries: {len(entry.get('memory_entries', []))}",
            f"  Registered: {entry.get('registered_at_utc', 'unknown')}",
        ]
        if entry.get("rollback_pointer"):
            lines.append(f"  Rollback: {entry['rollback_pointer']}")
        eval_results = entry.get("evaluation_results", {})
        if eval_results:
            lines.append(f"  Eval: {json.dumps(eval_results, indent=4)}")
        return "\n".join(lines)
