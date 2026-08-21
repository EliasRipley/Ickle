from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LANGUAGE_GLOB: dict[str, list[str]] = {
    "python": ["*.py"],
    "javascript": ["*.js", "*.mjs", "*.cjs"],
    "typescript": ["*.ts", "*.tsx"],
    "rust": ["*.rs"],
    "go": ["*.go"],
    "java": ["*.java"],
    "kotlin": ["*.kt", "*.kts"],
    "c": ["*.c", "*.h"],
    "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx"],
    "csharp": ["*.cs"],
    "ruby": ["*.rb"],
    "php": ["*.php"],
    "swift": ["*.swift"],
    "shell": ["*.sh", "*.bash", "*.zsh"],
    "toml": ["*.toml"],
    "yaml": ["*.yml", "*.yaml"],
    "json": ["*.json"],
    "markdown": ["*.md", "*.mdx"],
    "sql": ["*.sql"],
}

EXT_TO_LANG: dict[str, str] = {}
for lang, globs in LANGUAGE_GLOB.items():
    for g in globs:
        EXT_TO_LANG[g.lstrip("*")] = lang


@dataclass
class Symbol:
    name: str
    kind: str
    file_path: str
    line: int
    end_line: int
    parent: str = ""
    source: str = ""


@dataclass
class Reference:
    file_path: str
    line: int
    symbol_name: str
    context: str = ""


@dataclass
class FileInfo:
    path: str
    language: str
    size_bytes: int
    hash: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)


def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return EXT_TO_LANG.get(ext, "text")


def _file_hash(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return ""


def _parse_python_symbols(source: str, file_path: str) -> tuple[list[Symbol], list[str], list[Reference]]:
    symbols: list[Symbol] = []
    imports: list[str] = []
    references: list[Reference] = []

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return symbols, imports, references

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif node.module:
                imports.append(node.module)
        elif isinstance(node, ast.FunctionDef):
            symbols.append(Symbol(
                name=node.name,
                kind="function",
                file_path=file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                source=ast.get_source_segment(source, node) or "",
            ))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(
                name=node.name,
                kind="class",
                file_path=file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                source=ast.get_source_segment(source, node) or "",
            ))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            references.append(Reference(
                file_path=file_path,
                line=node.lineno,
                symbol_name=node.id,
            ))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append(Symbol(
                        name=target.id,
                        kind="variable",
                        file_path=file_path,
                        line=node.lineno,
                        end_line=node.lineno,
                        source=ast.get_source_segment(source, node) or "",
                    ))

    return symbols, imports, references


_RE_FUNCTION = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)",
    re.MULTILINE,
)
_RE_CLASS = re.compile(r"^\s*(?:export\s+)?class\s+(\w+)", re.MULTILINE)
_RE_IMPORT_JS = re.compile(r"(?:import\s+.*?from\s+['\"](.+?)['\"]|require\(['\"](.+?)['\"]\))", re.MULTILINE)
_RE_FN_RUST = re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)", re.MULTILINE)
_RE_STRUCT_RUST = re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)", re.MULTILINE)
_RE_FN_GO = re.compile(r"^\s*func\s+(?:\(.+?\)\s+)?(\w+)", re.MULTILINE)
_RE_TYPE_GO = re.compile(r"^\s*type\s+(\w+)\s+struct", re.MULTILINE)
_RE_FN_JAVA = re.compile(
    r"^\s*(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{",
    re.MULTILINE,
)
_RE_CLASS_JAVA = re.compile(r"^\s*(?:public|private|protected|static|\s)+class\s+(\w+)", re.MULTILINE)

GENERIC_PARSERS: dict[str, tuple[re.Pattern, re.Pattern | None]] = {
    "javascript": (_RE_FUNCTION, _RE_CLASS),
    "typescript": (_RE_FUNCTION, _RE_CLASS),
    "rust": (_RE_FN_RUST, _RE_STRUCT_RUST),
    "go": (_RE_FN_GO, _RE_TYPE_GO),
    "java": (_RE_FN_JAVA, _RE_CLASS_JAVA),
    "kotlin": (re.compile(r"^\s*(?:fun|class|object)\s+(\w+)", re.MULTILINE), None),
    "csharp": (re.compile(r"^\s*(?:void|int|string|bool|var|class)\s+(\w+)", re.MULTILINE), None),
    "ruby": (re.compile(r"^\s*def\s+(\w+)", re.MULTILINE), re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)),
    "php": (re.compile(r"^\s*function\s+(\w+)", re.MULTILINE), re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)),
}


def _parse_generic_symbols(source: str, file_path: str, language: str) -> tuple[list[Symbol], list[str]]:
    symbols: list[Symbol] = []
    imports: list[str] = []

    if language not in GENERIC_PARSERS:
        return symbols, imports

    fn_pat, type_pat = GENERIC_PARSERS[language]

    for match in fn_pat.finditer(source):
        name = match.group(1)
        if name and not name[0].isupper():
            line = source[: match.start()].count("\n") + 1
            symbols.append(Symbol(
                name=name,
                kind="function" if language != "kotlin" else "function_or_class",
                file_path=file_path,
                line=line,
                end_line=line,
            ))

    if type_pat:
        for match in type_pat.finditer(source):
            name = match.group(1)
            if name:
                line_num = source[: match.start()].count("\n") + 1
                symbols.append(Symbol(
                    name=name,
                    kind="class",
                    file_path=file_path,
                    line=line_num,
                    end_line=line_num,
                ))

    if language in ("javascript", "typescript"):
        for match in _RE_IMPORT_JS.finditer(source):
            imp = match.group(1) or match.group(2)
            if imp:
                imports.append(imp)

    return symbols, imports


class RepoIndex:
    def __init__(self, repo_root: str, cache_dir: str = "data/repo_index"):
        self.repo_root = Path(repo_root).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, FileInfo] = {}
        self._symbol_index: dict[str, list[Symbol]] = {}
        self._import_index: dict[str, list[str]] = {}

    @property
    def _cache_path(self) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9]", "_", str(self.repo_root))[:120]
        return self.cache_dir / f"{slug}.json"

    def _load_cache(self) -> dict[str, Any] | None:
        cache = self._cache_path
        if not cache.exists():
            return None
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("_version") != 2:
                return None
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, data: dict[str, Any]):
        data["_version"] = 2
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._cache_path)

    def _should_skip(self, rel: Path) -> bool:
        parts = rel.parts
        for skip in (".git", "node_modules", "__pycache__", ".venv", "venv", "target",
                     "build", "dist", ".tox", ".mypy_cache", ".pytest_cache",
                     ".next", ".nuxt", "vendor", ".cache"):
            if skip in parts:
                return True
        return False

    def index_repo(self, force: bool = False) -> int:
        cached = None if force else self._load_cache()
        self.files.clear()
        self._symbol_index.clear()
        self._import_index.clear()

        changed = 0
        for ext, lang in EXT_TO_LANG.items():
            for file_path in self.repo_root.rglob(f"*{ext}"):
                rel = str(file_path.relative_to(self.repo_root))
                if self._should_skip(file_path.relative_to(self.repo_root)):
                    continue
                fh = _file_hash(str(file_path))

                if cached and not force:
                    cached_file = cached.get(rel)
                    if isinstance(cached_file, dict) and cached_file.get("hash") == fh:
                        info = FileInfo(
                            path=rel,
                            language=cached_file.get("language", lang),
                            size_bytes=cached_file.get("size_bytes", 0),
                            hash=fh,
                        )
                        for s in cached_file.get("symbols", []):
                            info.symbols.append(Symbol(**s))
                        info.imports = cached_file.get("imports", [])
                        for r in cached_file.get("references", []):
                            info.references.append(Reference(**r))
                        self.files[rel] = info
                        for sym in info.symbols:
                            self._symbol_index.setdefault(sym.name, []).append(sym)
                        for imp in info.imports:
                            self._import_index.setdefault(imp, []).append(rel)
                        continue

                size = file_path.stat().st_size
                info = FileInfo(
                    path=rel,
                    language=lang,
                    size_bytes=size,
                    hash=fh,
                )

                if lang == "python":
                    try:
                        source = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        source = ""
                    symbols, imports, refs = _parse_python_symbols(source, rel)
                    info.symbols = symbols
                    info.imports = imports
                    info.references = refs
                else:
                    try:
                        source = file_path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        source = ""
                    symbols, imports = _parse_generic_symbols(source, rel, lang)
                    info.symbols = symbols
                    info.imports = imports

                self.files[rel] = info
                for sym in info.symbols:
                    self._symbol_index.setdefault(sym.name, []).append(sym)
                for imp in info.imports:
                    self._import_index.setdefault(imp, []).append(rel)
                changed += 1

        self._save_cache(self._serialize())
        return changed

    def _serialize(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for rel, info in self.files.items():
            out[rel] = {
                "path": info.path,
                "language": info.language,
                "size_bytes": info.size_bytes,
                "hash": info.hash,
                "symbols": [
                    {"name": s.name, "kind": s.kind, "file_path": s.file_path,
                     "line": s.line, "end_line": s.end_line, "parent": s.parent,
                     "source": s.source[:200] if s.source else ""}
                    for s in info.symbols
                ],
                "imports": info.imports,
                "references": [
                    {"file_path": r.file_path, "line": r.line,
                     "symbol_name": r.symbol_name, "context": r.context[:200] if r.context else ""}
                    for r in info.references
                ],
            }
        return out

    def search_symbol(self, name: str) -> list[Symbol]:
        if not self.files:
            self.index_repo()
        return self._symbol_index.get(name, [])

    def search_content(self, pattern: str, max_results: int = 50) -> list[dict[str, Any]]:
        if not self.files:
            self.index_repo()
        results: list[dict[str, Any]] = []
        compiled = re.compile(pattern, re.IGNORECASE)
        for rel, info in self.files.items():
            if len(results) >= max_results:
                break
            full_path = self.repo_root / rel
            try:
                text = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in compiled.finditer(text):
                if len(results) >= max_results:
                    break
                line_num = text[: match.start()].count("\n") + 1
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                context = text[start:end].replace("\n", " ").strip()
                results.append({
                    "file": rel,
                    "line": line_num,
                    "match": match.group(),
                    "context": context,
                })
        return results

    def get_file(self, rel_path: str) -> FileInfo | None:
        if not self.files:
            self.index_repo()
        return self.files.get(rel_path)

    def get_importers(self, module_name: str) -> list[str]:
        if not self.files:
            self.index_repo()
        return self._import_index.get(module_name, [])

    def get_dependencies(self, rel_path: str) -> list[str]:
        info = self.get_file(rel_path)
        if not info:
            return []
        return info.imports

    def find_definition(self, symbol_name: str) -> Symbol | None:
        results = self.search_symbol(symbol_name)
        if not results:
            return None
        defs = [s for s in results if s.kind in ("function", "class", "variable")]
        if defs:
            return defs[0]
        return results[0]

    def list_repo_structure(self, max_depth: int = 4) -> str:
        if not self.files:
            self.index_repo()
        lines: list[str] = []
        root_depth = len(self.repo_root.parts)
        dirs_seen: set[str] = set()
        for rel_str in sorted(self.files):
            p = Path(rel_str)
            depth = len(p.parts)
            if depth > max_depth:
                continue
            for i in range(1, depth):
                part = str(Path(*p.parts[:i]))
                if part not in dirs_seen:
                    lines.append(f"{'  ' * i}{p.parts[i - 1]}/")
                    dirs_seen.add(part)
            indent = "  " * depth
            lines.append(f"{indent}{p.name} ({self.files[rel_str].language})")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        if not self.files:
            self.index_repo()
        lang_count: dict[str, int] = {}
        total_symbols = 0
        for info in self.files.values():
            lang_count[info.language] = lang_count.get(info.language, 0) + 1
            total_symbols += len(info.symbols)
        return {
            "repo_root": str(self.repo_root),
            "total_files": len(self.files),
            "total_symbols": total_symbols,
            "languages": lang_count,
        }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build and query a repo symbol index")
    ap.add_argument("repo", help="Path to repository root")
    ap.add_argument("--action", default="index", choices=["index", "stats", "search", "structure", "symbols", "deps"])
    ap.add_argument("--symbol", help="Symbol name to search")
    ap.add_argument("--pattern", help="Content regex pattern to search")
    ap.add_argument("--file", help="Relative file path for dependency lookup")
    ap.add_argument("--force", action="store_true", help="Rebuild index from scratch")
    ap.add_argument("--cache-dir", default="data/repo_index")
    args = ap.parse_args()

    idx = RepoIndex(args.repo, cache_dir=args.cache_dir)

    if args.action == "index":
        n = idx.index_repo(force=args.force)
        print(f"Indexed {n} files (force={args.force})")
    elif args.action == "stats":
        import json
        print(json.dumps(idx.stats(), indent=2))
    elif args.action == "search" and args.pattern:
        for hit in idx.search_content(args.pattern):
            print(f"{hit['file']}:{hit['line']}  {hit['match']}  ...{hit['context'][:80]}...")
    elif args.action == "structure":
        print(idx.list_repo_structure())
    elif args.action == "symbols" and args.symbol:
        for sym in idx.search_symbol(args.symbol):
            print(f"{sym.file_path}:{sym.line}  {sym.kind} {sym.name}")
    elif args.action == "deps" and args.file:
        for dep in idx.get_dependencies(args.file):
            print(dep)


if __name__ == "__main__":
    main()
