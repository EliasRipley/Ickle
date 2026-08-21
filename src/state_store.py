import json
from pathlib import Path


class ILMStateStore:
    """Simple persistent local store for single-user ILM improvements/preferences."""

    def __init__(self, path: str = "data/ilm_state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"preferences": {}, "improvements": []})

    def _read(self) -> dict:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, payload: dict):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def set_preference(self, key: str, value: str):
        payload = self._read()
        payload.setdefault("preferences", {})[key] = value
        self._write(payload)

    def get_preference(self, key: str) -> str:
        return self._read().get("preferences", {}).get(key, "")

    def add_improvement_note(self, note: str):
        payload = self._read()
        improvements = payload.setdefault("improvements", [])
        improvements.append(note)
        improvements[:] = improvements[-200:]
        self._write(payload)

    def list_improvements(self) -> list[str]:
        return self._read().get("improvements", [])[-200:]
