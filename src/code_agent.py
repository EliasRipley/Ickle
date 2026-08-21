from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class ToolStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    FATAL = "fatal"


@dataclass
class ToolResult:
    tool: str
    status: ToolStatus
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditOp:
    file_path: str
    old_string: str
    new_string: str
    description: str = ""


@dataclass
class CodeSession:
    session_id: str
    workspace: str
    index_path: str = "data/repo_index"
    memory_path: str = "data/code_memory.json"
    max_tool_calls: int = 20
    max_edit_rounds: int = 5
    tool_calls_used: int = 0
    edits_applied: list[EditOp] = field(default_factory=list)
    test_history: list[dict[str, Any]] = field(default_factory=list)
    trajectory_log: list[dict[str, Any]] = field(default_factory=list)

    def _consume_tool(self):
        self.tool_calls_used += 1
        if self.tool_calls_used > self.max_tool_calls:
            raise RuntimeError(f"Tool-call budget exceeded ({self.max_tool_calls})")

    def _log_action(self, action: str, detail: dict[str, Any]):
        self.trajectory_log.append({
            "step": len(self.trajectory_log) + 1,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **detail,
        })


def read_file(file_path: str, start_line: int = 0, end_line: int = 0) -> str:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise OSError(f"Cannot read {file_path}: {e}")
    if start_line > 0 or end_line > 0:
        lines = text.splitlines()
        s = max(0, start_line - 1) if start_line else 0
        e = min(len(lines), end_line) if end_line else len(lines)
        text = "\n".join(lines[s:e])
    return text


def write_file(file_path: str, content: str) -> str:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    old_hash = ""
    if p.exists():
        old_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    p.write_text(content, encoding="utf-8")
    new_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return json.dumps({"path": str(p), "old_hash": old_hash, "new_hash": new_hash, "size": len(content)})


def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    content = p.read_text(encoding="utf-8")

    if replace_all:
        count = content.count(old_string)
        if count == 0:
            raise ValueError("old_string not found in content")
        new_content = content.replace(old_string, new_string)
        old_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        p.write_text(new_content, encoding="utf-8")
        new_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return json.dumps({
            "path": str(p),
            "old_hash": old_hash,
            "new_hash": new_hash,
            "replacements": count,
        })

    count = content.count(old_string)
    if count == 0:
        raise ValueError("old_string not found in content")
    if count > 1:
        raise ValueError(f"Found {count} matches for old_string. Provide more surrounding context or use replace_all.")
    new_content = content.replace(old_string, new_string, 1)
    old_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    p.write_text(new_content, encoding="utf-8")
    new_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return json.dumps({
        "path": str(p),
        "old_hash": old_hash,
        "new_hash": new_hash,
        "replacements": 1,
    })


def run_command(cmd: str, cwd: str = "", timeout_seconds: int = 120) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd or str(Path.cwd()),
        shell=True,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_seconds),
    )
    out: dict[str, Any] = {
        "returncode": result.returncode,
        "stdout": (result.stdout or "")[:8000],
        "stderr": (result.stderr or "")[:8000],
        "success": result.returncode == 0,
    }
    return json.dumps(out)


def run_tests(cmd: str = "pytest", cwd: str = "", timeout_seconds: int = 180) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        cwd=cwd or str(Path.cwd()),
        shell=True,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_seconds),
    )
    stdout = (result.stdout or "")
    stderr = (result.stderr or "")
    passed = re.findall(r"(\d+)\s+passed", stdout)
    failed = re.findall(r"(\d+)\s+failed", stdout)
    errors = re.findall(r"(\d+)\s+error", stdout)
    total_failures = int(failed[0]) if failed else 0
    total_errors = int(errors[0]) if errors else 0
    total_passed = int(passed[0]) if passed else 0
    return {
        "returncode": result.returncode,
        "passed": total_passed,
        "failed": total_failures,
        "errors": total_errors,
        "all_passed": result.returncode == 0 and total_failures == 0 and total_errors == 0,
        "stdout_tail": stdout[-3000:] if len(stdout) > 3000 else stdout,
        "stderr_tail": stderr[-2000:] if len(stderr) > 2000 else stderr,
    }


def run_lint(file_path: str, cwd: str = "") -> dict[str, Any]:
    p = Path(file_path)
    cmd = ""
    suffix = p.suffix.lower()

    if suffix == ".py":
        cmd = f"{sys.executable} -m flake8 --max-line-length=120 {p.resolve()}"
    elif suffix in (".js", ".ts"):
        cmd = f"npx eslint {p.resolve()}"
    elif suffix == ".rs":
        cmd = f"cargo clippy --manifest-path {Path(cwd or '.') / 'Cargo.toml'}"

    if not cmd:
        return {"linted": True, "tool": "none", "message": f"No linter available for {suffix}"}

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, cwd=cwd or str(Path.cwd()))
        issues = len((result.stdout or "").strip().splitlines()) + len((result.stderr or "").strip().splitlines())
        return {
            "linted": True,
            "tool": cmd.split()[0] if cmd else "unknown",
            "passed": result.returncode == 0,
            "issues": issues,
            "output": (result.stdout or "")[:2000],
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"linted": False, "error": str(e)}


def run_typecheck(cwd: str = "") -> dict[str, Any]:
    cwd_path = Path(cwd or ".")
    if (cwd_path / "pyproject.toml").exists() or (cwd_path / "mypy.ini").exists() or (cwd_path / "setup.cfg").exists():
        cmd = f"{sys.executable} -m mypy {cwd_path} --no-error-summary"
    else:
        return {"typechecked": True, "tool": "none", "message": "No mypy config found"}

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=str(cwd_path))
        issues = len((result.stdout or "").strip().splitlines())
        return {
            "typechecked": True,
            "tool": "mypy",
            "passed": result.returncode == 0,
            "issues": issues,
            "output": (result.stdout or "")[:2000],
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {"typechecked": False, "error": str(e)}


def search_repo(pattern: str, repo_root: str = "", file_types: str = "*") -> list[dict[str, Any]]:
    root = Path(repo_root) if repo_root else Path.cwd()
    results: list[dict[str, Any]] = []
    compiled = re.compile(pattern, re.IGNORECASE)
    ext_patterns: list[str] = []
    if file_types != "*":
        ext_patterns = [f"*.{e.strip()}" for e in file_types.split(",") if e.strip()]

    for file_path in root.rglob("*"):
        if file_path.is_dir():
            continue
        if any(skip in file_path.parts for skip in (".git", "node_modules", "__pycache__", ".venv", "venv", "target")):
            continue
        if ext_patterns and not any(file_path.match(ep) for ep in ext_patterns):
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in compiled.finditer(text):
            if len(results) >= 50:
                break
            line_num = text[: match.start()].count("\n") + 1
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace("\n", " ").strip()
            results.append({
                "file": str(file_path.relative_to(root)),
                "line": line_num,
                "match": match.group(),
                "context": context,
            })
        if len(results) >= 50:
            break
    return results


class CodeAgent:
    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace).resolve()
        self.sessions: dict[str, CodeSession] = {}
        self._default_test_cmd = "pytest"
        self._auto_detect_test_cmd()

    def _auto_detect_test_cmd(self):
        root = self.workspace
        if (root / "pyproject.toml").exists():
            self._default_test_cmd = "pytest"
        elif (root / "package.json").exists():
            pkg = json.loads(Path(root / "package.json").read_text())
            test_script = (pkg.get("scripts") or {}).get("test", "")
            if "vitest" in test_script:
                self._default_test_cmd = "npx vitest run"
            elif "jest" in test_script:
                self._default_test_cmd = "npx jest"
            elif "mocha" in test_script:
                self._default_test_cmd = "npx mocha"
            else:
                self._default_test_cmd = "npm test"
        elif (root / "Cargo.toml").exists():
            self._default_test_cmd = "cargo test"

    def new_session(self, session_id: str = "") -> CodeSession:
        sid = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]
        session = CodeSession(session_id=sid, workspace=str(self.workspace))
        self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> CodeSession | None:
        return self.sessions.get(session_id)

    def read(self, file_path: str, start_line: int = 0, end_line: int = 0) -> ToolResult:
        try:
            content = read_file(file_path, start_line, end_line)
            return ToolResult(
                tool="read_file",
                status=ToolStatus.SUCCESS,
                output=content[:20000],
                metadata={"file": file_path, "lines": len(content.splitlines())},
            )
        except (FileNotFoundError, OSError) as e:
            return ToolResult(tool="read_file", status=ToolStatus.FAILED, error=str(e))

    def write(self, file_path: str, content: str, session: CodeSession) -> ToolResult:
        try:
            session._consume_tool()
            meta = json.loads(write_file(file_path, content))
            session._log_action("write_file", {"file": file_path})
            return ToolResult(
                tool="write_file",
                status=ToolStatus.SUCCESS,
                output=f"Written {meta['size']} bytes to {file_path}",
                metadata=meta,
            )
        except (FileNotFoundError, OSError, ValueError) as e:
            return ToolResult(tool="write_file", status=ToolStatus.FAILED, error=str(e))

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False, description: str = "", session: CodeSession | None = None) -> ToolResult:
        try:
            if session:
                session._consume_tool()
            meta = json.loads(edit_file(file_path, old_string, new_string, replace_all))
            if session:
                session.edits_applied.append(EditOp(
                    file_path=file_path,
                    old_string=old_string[:200],
                    new_string=new_string[:200],
                    description=description or "edit",
                ))
                session._log_action("edit_file", {"file": file_path, "description": description or "edit"})
            return ToolResult(
                tool="edit_file",
                status=ToolStatus.SUCCESS,
                output=f"Applied {meta['replacements']} replacement(s) to {file_path}",
                metadata=meta,
            )
        except (FileNotFoundError, OSError, ValueError) as e:
            return ToolResult(tool="edit_file", status=ToolStatus.FAILED, error=str(e))

    def run(self, cmd: str, cwd: str = "", timeout_seconds: int = 120, session: CodeSession | None = None) -> ToolResult:
        try:
            if session:
                session._consume_tool()
            meta = json.loads(run_command(cmd, cwd, timeout_seconds))
            rc = meta.get("returncode", -1)
            status = ToolStatus.SUCCESS if rc == 0 else ToolStatus.FAILED
            result = ToolResult(
                tool="run_command",
                status=status,
                output=meta.get("stdout", "")[:5000],
                error=meta.get("stderr", "")[:5000],
                metadata=meta,
            )
            if session:
                session._log_action("run_command", {"cmd": cmd[:200], "success": rc == 0})
            return result
        except (subprocess.TimeoutExpired, OSError) as e:
            if session:
                session._consume_tool()
            return ToolResult(tool="run_command", status=ToolStatus.FAILED, error=str(e))

    def run_tests(self, cmd: str = "", session: CodeSession | None = None) -> ToolResult:
        try:
            if session:
                session._consume_tool()
            test_cmd = cmd or self._default_test_cmd
            result = run_tests(test_cmd, cwd=str(self.workspace))
            all_pass = result.get("all_passed", False)
            if session:
                session.test_history.append({"command": test_cmd, "all_passed": all_pass, "timestamp": datetime.now(timezone.utc).isoformat()})
                session._log_action("run_tests", {"passed": all_pass})
            return ToolResult(
                tool="run_tests",
                status=ToolStatus.SUCCESS if all_pass else ToolStatus.RETRY,
                output=json.dumps(result, indent=2),
                metadata=result,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            if session:
                session._consume_tool()
            return ToolResult(tool="run_tests", status=ToolStatus.FAILED, error=str(e))

    def lint(self, file_path: str, session: CodeSession | None = None) -> ToolResult:
        try:
            if session:
                session._consume_tool()
            result = run_lint(file_path, cwd=str(self.workspace))
            passed = result.get("passed", False)
            if session:
                session._log_action("lint", {"file": file_path, "passed": passed})
            return ToolResult(
                tool="lint",
                status=ToolStatus.SUCCESS if passed else ToolStatus.RETRY,
                output=json.dumps(result, indent=2),
                metadata=result,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            if session:
                session._consume_tool()
            return ToolResult(tool="lint", status=ToolStatus.FAILED, error=str(e))

    def typecheck(self, session: CodeSession | None = None) -> ToolResult:
        try:
            if session:
                session._consume_tool()
            result = run_typecheck(str(self.workspace))
            passed = result.get("passed", False)
            if session:
                session._log_action("typecheck", {"passed": passed})
            return ToolResult(
                tool="typecheck",
                status=ToolStatus.SUCCESS if passed else ToolStatus.RETRY,
                output=json.dumps(result, indent=2),
                metadata=result,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            if session:
                session._consume_tool()
            return ToolResult(tool="typecheck", status=ToolStatus.FAILED, error=str(e))

    def search(self, pattern: str, file_types: str = "*") -> ToolResult:
        try:
            hits = search_repo(pattern, str(self.workspace), file_types)
            return ToolResult(
                tool="search_repo",
                status=ToolStatus.SUCCESS,
                output=json.dumps(hits, indent=2)[:10000],
                metadata={"pattern": pattern, "count": len(hits)},
            )
        except Exception as e:
            return ToolResult(tool="search_repo", status=ToolStatus.FAILED, error=str(e))

    def validate_edits(self, session: CodeSession) -> ToolResult:
        issues: list[str] = []
        for edit in session.edits_applied:
            p = Path(edit.file_path)
            if not p.exists():
                issues.append(f"Missing file: {edit.file_path}")
        if issues:
            return ToolResult(tool="validate_edits", status=ToolStatus.FAILED, error="\n".join(issues))
        return ToolResult(tool="validate_edits", status=ToolStatus.SUCCESS, output=f"All {len(session.edits_applied)} edits valid")

    def summarize_session(self, session: CodeSession) -> str:
        lines = [
            f"Session: {session.session_id}",
            f"Workspace: {session.workspace}",
            f"Tool calls used: {session.tool_calls_used}/{session.max_tool_calls}",
            f"Edits applied: {len(session.edits_applied)}",
            f"Test runs: {len(session.test_history)}",
        ]
        if session.test_history:
            last = session.test_history[-1]
            lines.append(f"Last test: {'PASSED' if last.get('all_passed') else 'FAILED'}")
        if session.trajectory_log:
            lines.append(f"Trajectory steps: {len(session.trajectory_log)}")
        return "\n".join(lines)

    def export_trajectory(self, session: CodeSession, out_path: str) -> int:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for step in session.trajectory_log:
                f.write(json.dumps(step, ensure_ascii=False) + "\n")
        return len(session.trajectory_log)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Code agent for deterministic code editing workflows")
    ap.add_argument("--workspace", default=".", help="Project workspace root")
    ap.add_argument("--read", help="Read a file")
    ap.add_argument("--search", help="Search repo for pattern")
    ap.add_argument("--file-types", default="*", help="File type filter for search")
    ap.add_argument("--run", help="Run a shell command")
    ap.add_argument("--test", action="store_true", help="Run test suite")
    ap.add_argument("--lint", help="Lint a specific file")
    ap.add_argument("--typecheck", action="store_true", help="Run type checker")
    ap.add_argument("--edit-file", help="File to edit")
    ap.add_argument("--old-string", help="Text to replace")
    ap.add_argument("--new-string", help="Replacement text")
    ap.add_argument("--replace-all", action="store_true", help="Replace all occurrences")
    ap.add_argument("--write-file", help="File to write")
    ap.add_argument("--content", help="Content to write")
    ap.add_argument("--session", default="", help="Session ID")
    ap.add_argument("--export-trajectory", help="Export session trajectory to JSONL")
    args = ap.parse_args()

    agent = CodeAgent(workspace=args.workspace)
    session = agent.new_session(args.session) if args.session else None

    if args.read:
        result = agent.read(args.read)
        print(result.output or result.error)

    elif args.search:
        result = agent.search(args.search, args.file_types)
        print(result.output or result.error)

    elif args.run:
        result = agent.run(args.run)
        print(result.output or result.error)

    elif args.test:
        result = agent.run_tests(session=session or agent.new_session())
        print(result.output)

    elif args.lint:
        result = agent.lint(args.lint, session=session)
        print(result.output)

    elif args.typecheck:
        result = agent.typecheck(session=session)
        print(result.output)

    elif args.edit_file and args.old_string and args.new_string:
        result = agent.edit(
            args.edit_file, args.old_string, args.new_string,
            replace_all=args.replace_all, session=session,
        )
        print(result.output or result.error)

    elif args.write_file and args.content:
        result = agent.write(args.write_file, args.content, session or agent.new_session())
        print(result.output or result.error)

    elif args.export_trajectory and session:
        n = agent.export_trajectory(session, args.export_trajectory)
        print(f"Exported {n} trajectory steps to {args.export_trajectory}")

    elif session:
        print(agent.summarize_session(session))

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
