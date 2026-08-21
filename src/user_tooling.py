from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class ToolPermissionError(RuntimeError):
    pass


_NETWORK_MODULES = {
    "requests", "httpx", "socket", "urllib", "urllib2", "urllib3",
    "aiohttp", "http", "ftplib", "smtplib", "telnetlib", "websockets",
}
_PROCESS_MODULES = {"subprocess", "shlex", "pty", "multiprocessing", "asyncio"}
_PROCESS_CALL_ATTRS = {"system", "popen", "spawnl", "spawnv", "spawnve", "execl", "execv"}
_DYNAMIC_IMPORT_NAMES = {"__import__", "import_module"}


def _collect_static_imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _collect_dynamic_risk(tree: ast.AST) -> tuple[bool, bool]:
    """Flags network/process risk from patterns a plain import scan misses:
    __import__("socket"), importlib.import_module("subprocess"), os.system(...),
    os.popen(...), and eval/exec (conservatively treated as both, since their
    contents can't be statically analyzed at all)."""
    network_risk = False
    process_risk = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in ("eval", "exec"):
            network_risk = True
            process_risk = True
        elif name in _DYNAMIC_IMPORT_NAMES:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    mod = arg.value.split(".")[0]
                    if mod in _NETWORK_MODULES:
                        network_risk = True
                    if mod in _PROCESS_MODULES:
                        process_risk = True
        elif name in _PROCESS_CALL_ATTRS and isinstance(func, ast.Attribute):
            process_risk = True
    return network_risk, process_risk


@dataclass
class ToolPermissions:
    filesystem_read: bool = True
    filesystem_write: bool = False
    network: bool = False
    process: bool = False

    @staticmethod
    def from_dict(payload: dict) -> "ToolPermissions":
        return ToolPermissions(
            filesystem_read=bool(payload.get("filesystem_read", True)),
            filesystem_write=bool(payload.get("filesystem_write", False)),
            network=bool(payload.get("network", False)),
            process=bool(payload.get("process", False)),
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "filesystem_read": self.filesystem_read,
            "filesystem_write": self.filesystem_write,
            "network": self.network,
            "process": self.process,
        }


class UserToolRegistry:
    """Loads user-owned tool plugins from local filesystem.

    Plugin contract:
    - file path: user_tools/<tool_name>.py
    - function: run(payload: dict) -> str
    """

    def __init__(self, tools_dir: str = "user_tools", default_timeout_seconds: int = 20):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.default_timeout_seconds = max(1, int(default_timeout_seconds))

    def list_tools(self) -> list[str]:
        return sorted(p.stem for p in self.tools_dir.glob("*.py") if p.is_file())

    def _manifest_path(self, tool_name: str) -> Path:
        return self.tools_dir / f"{tool_name}.tool.json"

    def _load_permissions(self, tool_name: str) -> ToolPermissions:
        manifest_path = self._manifest_path(tool_name)
        if not manifest_path.exists():
            return ToolPermissions()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ToolPermissionError(f"Invalid tool manifest JSON for '{tool_name}': {exc}") from exc
        perms = payload.get("permissions", {})
        if not isinstance(perms, dict):
            raise ToolPermissionError(f"Tool manifest for '{tool_name}' has invalid permissions block.")
        return ToolPermissions.from_dict(perms)

    def _enforce_static_policy(self, tool_path: Path, permissions: ToolPermissions):
        # AST-based, not a substring scan: a plain text search for "import
        # requests" is trivially defeated by whitespace tricks
        # ("import  requests"), dynamic imports (__import__("requests")),
        # or aliasing -- and just as easily false-positives on a string
        # literal or comment that happens to contain that text. Parsing the
        # real syntax tree checks actual import statements and calls.
        source = tool_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ToolPermissionError(f"Tool '{tool_path.stem}' has invalid Python syntax: {exc}") from exc

        imported = _collect_static_imports(tree)
        dyn_network, dyn_process = _collect_dynamic_risk(tree)

        if not permissions.network and (dyn_network or imported & _NETWORK_MODULES):
            raise ToolPermissionError(
                f"Tool '{tool_path.stem}' imports network modules but network permission is disabled."
            )

        if not permissions.process and (dyn_process or imported & _PROCESS_MODULES):
            raise ToolPermissionError(
                f"Tool '{tool_path.stem}' imports process execution modules but process permission is disabled."
            )

    def describe_tool_permissions(self, name: str) -> dict[str, bool]:
        tool_path = self.tools_dir / f"{name}.py"
        if not tool_path.exists():
            available = ", ".join(self.list_tools())
            raise FileNotFoundError(f"Tool '{name}' not found. Available: {available}")
        perms = self._load_permissions(name)
        return perms.as_dict()

    def run_tool(self, name: str, payload_json: str = "{}") -> str:
        tool_path = self.tools_dir / f"{name}.py"
        if not tool_path.exists():
            available = ", ".join(self.list_tools())
            raise FileNotFoundError(f"Tool '{name}' not found. Available: {available}")

        permissions = self._load_permissions(name)
        self._enforce_static_policy(tool_path, permissions)

        payload = json.loads(payload_json) if payload_json.strip() else {}
        command = [
            sys.executable,
            "-m",
            "src.user_tool_runner",
            "--tool-file",
            str(tool_path.resolve()),
            "--payload-json",
            json.dumps(payload, ensure_ascii=False),
        ]
        env = dict(os.environ)
        env["ICKLE_TOOL_PERMISSIONS"] = json.dumps(permissions.as_dict(), ensure_ascii=False)
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.default_timeout_seconds,
            env=env,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or f"exit={proc.returncode}"
            raise RuntimeError(f"Tool '{name}' failed: {detail}")
        raw = (proc.stdout or "").strip()
        if not raw:
            return ""
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(output, dict) and output.get("ok") is True:
            return str(output.get("result", ""))
        if isinstance(output, dict) and output.get("error"):
            raise RuntimeError(str(output["error"]))
        return str(output)
