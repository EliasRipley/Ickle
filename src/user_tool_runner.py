from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_tool_module(tool_file: Path):
    spec = importlib.util.spec_from_file_location(f"user_tool_{tool_file.stem}", tool_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tool module: {tool_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_tool(tool_file: Path, payload: dict[str, Any]) -> str:
    module = _load_tool_module(tool_file)
    if not hasattr(module, "run"):
        raise AttributeError(f"Tool '{tool_file.stem}' must export run(payload: dict) -> str")
    result = module.run(payload)
    return str(result)


def main():
    parser = argparse.ArgumentParser(description="Isolated user tool runner for Ickle.")
    parser.add_argument("--tool-file", required=True)
    parser.add_argument("--payload-json", default="{}")
    args = parser.parse_args()

    tool_file = Path(args.tool_file).resolve()
    try:
        payload = json.loads(args.payload_json) if str(args.payload_json).strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("payload-json must decode to an object")
        out = _run_tool(tool_file, payload)
        print(json.dumps({"ok": True, "result": out}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

