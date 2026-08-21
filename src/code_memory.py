from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_EDITS = 2000


@dataclass
class ProjectContext:
    name: str
    root: str
    language: str = ""
    conventions: list[str] = field(default_factory=list)
    architecture_notes: list[str] = field(default_factory=list)


@dataclass
class FileContext:
    path: str
    purpose: str = ""
    key_symbols: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    last_tested: str = ""
    test_command: str = ""


@dataclass
class EditRecord:
    file_path: str
    timestamp: str
    description: str
    old_hash: str = ""
    new_hash: str = ""
    tests_passed: bool = False
    lint_passed: bool = False
    errors: list[str] = field(default_factory=list)


class CodeMemory:
    def __init__(self, storage_path: str = "data/code_memory.json"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            self._data = {"projects": {}, "files": {}, "edits": [], "conventions": {}}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {"projects": {}, "files": {}, "edits": [], "conventions": {}}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def register_project(self, name: str, root: str, language: str = "") -> ProjectContext:
        proj = ProjectContext(name=name, root=root, language=language)
        self._data.setdefault("projects", {})[name] = asdict(proj)
        self._save()
        return proj

    def get_project(self, name: str) -> ProjectContext | None:
        raw = self._data.get("projects", {}).get(name)
        if not raw:
            return None
        return ProjectContext(**raw)

    def list_projects(self) -> list[str]:
        return list(self._data.get("projects", {}).keys())

    def set_file_context(self, file_path: str, purpose: str = "", key_symbols: list[str] | None = None,
                         dependencies: list[str] | None = None, test_command: str = ""):
        existing = self._data.setdefault("files", {}).get(file_path, {})
        fc = FileContext(
            path=file_path,
            purpose=purpose or existing.get("purpose", ""),
            key_symbols=key_symbols or existing.get("key_symbols", []),
            dependencies=dependencies or existing.get("dependencies", []),
            dependents=existing.get("dependents", []),
            last_tested=existing.get("last_tested", ""),
            test_command=test_command or existing.get("test_command", ""),
        )
        self._data["files"][file_path] = asdict(fc)
        for dep in fc.dependencies:
            dep_data = self._data["files"].setdefault(dep, asdict(FileContext(path=dep)))
            if file_path not in dep_data.get("dependents", []):
                dep_data.setdefault("dependents", []).append(file_path)
        self._save()

    def get_file_context(self, file_path: str) -> FileContext | None:
        raw = self._data.get("files", {}).get(file_path)
        if not raw:
            return None
        return FileContext(**raw)

    def record_edit(self, file_path: str, description: str, old_hash: str = "",
                    new_hash: str = "", tests_passed: bool = False,
                    lint_passed: bool = False, errors: list[str] | None = None):
        now = datetime.now(timezone.utc).isoformat()
        rec = EditRecord(
            file_path=file_path,
            timestamp=now,
            description=description,
            old_hash=old_hash,
            new_hash=new_hash,
            tests_passed=tests_passed,
            lint_passed=lint_passed,
            errors=list(errors or []),
        )
        self._data.setdefault("edits", []).append(asdict(rec))
        if len(self._data["edits"]) > MAX_EDITS:
            self._data["edits"][:] = self._data["edits"][-MAX_EDITS:]
        if tests_passed and file_path in self._data.get("files", {}):
            self._data["files"][file_path]["last_tested"] = now
        self._save()

    def recent_edits(self, file_path: str | None = None, limit: int = 20) -> list[EditRecord]:
        all_edits = [EditRecord(**r) for r in self._data.get("edits", [])]
        if file_path:
            all_edits = [e for e in all_edits if e.file_path == file_path]
        return list(reversed(all_edits))[:limit]

    def add_convention(self, project: str, convention: str):
        proj = self._data.setdefault("conventions", {}).setdefault(project, [])
        if convention not in proj:
            proj.append(convention)
        self._save()

    def get_conventions(self, project: str) -> list[str]:
        return list(self._data.get("conventions", {}).get(project, []))

    def add_architecture_note(self, project: str, note: str):
        proj = self._data.get("projects", {}).get(project, {})
        notes = proj.setdefault("architecture_notes", [])
        if note not in notes:
            notes.append(note)
        self._save()

    def get_architecture_notes(self, project: str) -> list[str]:
        proj = self._data.get("projects", {}).get(project, {})
        return list(proj.get("architecture_notes", []))

    def mark_tested(self, file_path: str):
        now = datetime.now(timezone.utc).isoformat()
        if file_path in self._data.get("files", {}):
            self._data["files"][file_path]["last_tested"] = now
        else:
            self._data.setdefault("files", {})[file_path] = asdict(FileContext(path=file_path, last_tested=now))
        self._save()

    def get_project_summary(self, name: str) -> str:
        proj = self.get_project(name)
        if not proj:
            return f"No project named '{name}'"
        files = [
            k for k, v in self._data.get("files", {}).items()
            if v.get("path", "").startswith(proj.root) or k.startswith(proj.root)
        ]
        conventions = self.get_conventions(name)
        notes = self.get_architecture_notes(name)
        edits = [e for e in self._data.get("edits", []) if proj.root in e.get("file_path", "")]
        lines = [
            f"Project: {name}",
            f"Root: {proj.root}",
            f"Language: {proj.language or 'unknown'}",
            f"Tracked files: {len(files)}",
            f"Tracked edits: {len(edits)}",
        ]
        if conventions:
            lines.append("Conventions:")
            for c in conventions:
                lines.append(f"  - {c}")
        if notes:
            lines.append("Architecture notes:")
            for n in notes[-10:]:
                lines.append(f"  - {n}")
        return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Manage code memory for a project")
    ap.add_argument("project", help="Project name")
    ap.add_argument("--root", help="Project root directory")
    ap.add_argument("--language", default="")
    ap.add_argument("--register", action="store_true", help="Register a new project")
    ap.add_argument("--summary", action="store_true", help="Show project summary")
    ap.add_argument("--file", help="Record file context")
    ap.add_argument("--purpose", help="File purpose description")
    ap.add_argument("--deps", help="Comma-separated file dependencies")
    ap.add_argument("--convention", help="Add a project convention")
    ap.add_argument("--note", help="Add an architecture note")
    ap.add_argument("--edits", type=int, default=0, help="Show recent edits (limit)")
    ap.add_argument("--storage", default="data/code_memory.json")
    args = ap.parse_args()

    cm = CodeMemory(storage_path=args.storage)

    if args.register and args.root:
        p = cm.register_project(args.project, args.root, args.language)
        print(f"Registered project '{args.project}' at {args.root}")

    if args.summary:
        print(cm.get_project_summary(args.project))

    if args.file:
        deps = [d.strip() for d in (args.deps or "").split(",") if d.strip()]
        cm.set_file_context(
            file_path=args.file,
            purpose=args.purpose or "",
            dependencies=deps,
        )
        print(f"Recorded context for {args.file}")

    if args.convention:
        cm.add_convention(args.project, args.convention)
        print(f"Added convention to {args.project}: {args.convention}")

    if args.note:
        cm.add_architecture_note(args.project, args.note)
        print(f"Added note to {args.project}")

    if args.edits > 0:
        for e in cm.recent_edits(limit=args.edits):
            status = "PASS" if e.tests_passed else ("LINT" if e.lint_passed else "FAIL")
            print(f"[{e.timestamp[:19]}] {status} {e.file_path}: {e.description[:80]}")


if __name__ == "__main__":
    main()
