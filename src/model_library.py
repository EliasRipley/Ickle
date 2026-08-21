from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelLibraryError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "model-package"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_entry(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path.replace("\\", "/"),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


class ModelLibrary:
    def __init__(self, root: str = "data/model_library"):
        self.root = Path(root)
        self.packages_dir = self.root / "packages"
        self.index_path = self.root / "library_index.json"
        self._ensure_layout()

    def _ensure_layout(self):
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._save_index({"schema_version": "1.0", "updated_at_utc": _utc_now(), "packages": []})

    def _load_index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            payload = {"schema_version": "1.0", "updated_at_utc": _utc_now(), "packages": []}
        if not isinstance(payload, dict):
            payload = {"schema_version": "1.0", "updated_at_utc": _utc_now(), "packages": []}
        if "packages" not in payload or not isinstance(payload["packages"], list):
            payload["packages"] = []
        if "schema_version" not in payload:
            payload["schema_version"] = "1.0"
        if "updated_at_utc" not in payload:
            payload["updated_at_utc"] = _utc_now()
        return payload

    def _save_index(self, payload: dict[str, Any]):
        payload["updated_at_utc"] = _utc_now()
        self.index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _upsert_package_record(self, record: dict[str, Any]):
        index = self._load_index()
        packages = index["packages"]
        package_id = str(record["package_id"])
        replaced = False
        for i, existing in enumerate(packages):
            if str(existing.get("package_id")) == package_id:
                packages[i] = record
                replaced = True
                break
        if not replaced:
            packages.append(record)
        index["packages"] = packages
        self._save_index(index)

    def list_packages(self) -> list[dict[str, Any]]:
        rows = self._load_index()["packages"]
        rows = [row for row in rows if isinstance(row, dict)]
        rows.sort(key=lambda row: str(row.get("created_at_utc", "")), reverse=True)
        return rows

    def _find_package_record(self, package_id: str) -> dict[str, Any]:
        for row in self.list_packages():
            if str(row.get("package_id")) == package_id:
                return row
        raise ModelLibraryError(f"Package '{package_id}' not found in library index.")

    def validate_package(self, package_dir: str | Path) -> dict[str, Any]:
        root = Path(package_dir).resolve()
        errors: list[str] = []
        warnings: list[str] = []
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            return {
                "valid": False,
                "package_dir": str(root),
                "errors": ["manifest.json not found"],
                "warnings": [],
                "manifest": None,
            }

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {
                "valid": False,
                "package_dir": str(root),
                "errors": [f"Failed to parse manifest.json: {exc}"],
                "warnings": [],
                "manifest": None,
            }

        required_fields = ["schema_version", "package_id", "name", "version", "author", "files", "entrypoint"]
        for field in required_fields:
            if field not in manifest:
                errors.append(f"manifest missing required field '{field}'")

        files = manifest.get("files", [])
        if not isinstance(files, list) or not files:
            errors.append("manifest.files must be a non-empty list")
            files = []

        for file_entry in files:
            if not isinstance(file_entry, dict):
                errors.append("manifest.files contains non-object entry")
                continue
            rel = str(file_entry.get("path", "")).replace("\\", "/")
            sha = str(file_entry.get("sha256", "")).lower()
            size = int(file_entry.get("size_bytes", -1))
            if not rel:
                errors.append("manifest file entry missing path")
                continue
            target = root / rel
            if not target.exists() or not target.is_file():
                errors.append(f"manifest file missing on disk: {rel}")
                continue
            if size >= 0 and int(target.stat().st_size) != size:
                errors.append(f"size mismatch for {rel}")
            actual_sha = _sha256_file(target)
            if sha and actual_sha != sha:
                errors.append(f"sha256 mismatch for {rel}")

        entrypoint = str(manifest.get("entrypoint", "")).replace("\\", "/")
        if entrypoint:
            entrypoint_path = root / entrypoint
            if not entrypoint_path.exists():
                errors.append(f"entrypoint file missing: {entrypoint}")
        else:
            warnings.append("entrypoint is empty")

        return {
            "valid": not errors,
            "package_dir": str(root),
            "errors": errors,
            "warnings": warnings,
            "manifest": manifest,
        }

    def export_package(
        self,
        *,
        model_path: str,
        name: str,
        version: str,
        author: str,
        summary: str = "",
        tags: list[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        source_model = Path(model_path).resolve()
        if not source_model.exists():
            raise ModelLibraryError(f"Model file does not exist: {source_model}")
        if not source_model.is_file():
            raise ModelLibraryError(f"Model path is not a file: {source_model}")

        package_id = _slugify(f"{name}-{version}")
        package_dir = self.packages_dir / package_id
        if package_dir.exists():
            if not overwrite:
                raise ModelLibraryError(f"Package already exists: {package_id}. Use --overwrite to replace it.")
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)

        model_filename = source_model.name
        exported_model = package_dir / model_filename
        shutil.copyfile(source_model, exported_model)

        files: list[dict[str, Any]] = [_file_entry(exported_model, model_filename)]
        exported_meta_name: str | None = None
        source_meta = Path(str(source_model) + ".meta.json")
        if source_meta.exists() and source_meta.is_file():
            exported_meta_name = source_meta.name
            exported_meta = package_dir / exported_meta_name
            shutil.copyfile(source_meta, exported_meta)
            files.append(_file_entry(exported_meta, exported_meta_name))

        manifest = {
            "schema_version": "1.0",
            "package_id": package_id,
            "name": name.strip(),
            "version": version.strip(),
            "author": author.strip(),
            "summary": summary.strip(),
            "tags": [t.strip() for t in (tags or []) if str(t).strip()],
            "created_at_utc": _utc_now(),
            "entrypoint": model_filename,
            "meta_file": exported_meta_name,
            "files": files,
        }
        manifest_path = package_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        record = {
            "package_id": package_id,
            "name": manifest["name"],
            "version": manifest["version"],
            "author": manifest["author"],
            "summary": manifest["summary"],
            "tags": manifest["tags"],
            "created_at_utc": manifest["created_at_utc"],
            "entrypoint": model_filename,
            "local_path": str(package_dir.resolve()),
            "model_sha256": _sha256_file(exported_model),
            "model_size_bytes": int(exported_model.stat().st_size),
        }
        self._upsert_package_record(record)
        return {
            "package_id": package_id,
            "package_dir": str(package_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
        }

    def import_package(self, package_dir: str, *, overwrite: bool = False) -> dict[str, Any]:
        source_dir = Path(package_dir).resolve()
        report = self.validate_package(source_dir)
        if not report["valid"]:
            raise ModelLibraryError("Invalid package: " + "; ".join(report["errors"]))

        manifest = report["manifest"]
        package_id = str(manifest["package_id"])
        target_dir = self.packages_dir / package_id
        if target_dir.exists():
            if not overwrite:
                raise ModelLibraryError(f"Package already exists: {package_id}. Use --overwrite to replace it.")
            shutil.rmtree(target_dir)

        shutil.copytree(source_dir, target_dir)
        copied_manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
        entrypoint = str(copied_manifest["entrypoint"])
        model_path = target_dir / entrypoint
        record = {
            "package_id": package_id,
            "name": str(copied_manifest.get("name", package_id)),
            "version": str(copied_manifest.get("version", "unknown")),
            "author": str(copied_manifest.get("author", "unknown")),
            "summary": str(copied_manifest.get("summary", "")),
            "tags": list(copied_manifest.get("tags", [])),
            "created_at_utc": str(copied_manifest.get("created_at_utc", _utc_now())),
            "entrypoint": entrypoint,
            "local_path": str(target_dir.resolve()),
            "model_sha256": _sha256_file(model_path),
            "model_size_bytes": int(model_path.stat().st_size),
        }
        self._upsert_package_record(record)
        return {"package_id": package_id, "package_dir": str(target_dir.resolve())}

    def install_package(self, package_id: str, out_path: str | None = None, *, overwrite: bool = False) -> dict[str, Any]:
        record = self._find_package_record(package_id)
        package_dir = Path(str(record["local_path"])).resolve()
        report = self.validate_package(package_dir)
        if not report["valid"]:
            raise ModelLibraryError("Package failed validation before install: " + "; ".join(report["errors"]))

        manifest = report["manifest"]
        entrypoint = str(manifest["entrypoint"]).replace("\\", "/")
        source_model = package_dir / entrypoint
        suffix = source_model.suffix or ".pt"
        target = Path(out_path).resolve() if out_path else (Path("models") / f"{package_id}{suffix}").resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise ModelLibraryError(f"Target model already exists: {target}. Use --overwrite to replace it.")

        shutil.copyfile(source_model, target)
        source_meta = package_dir / f"{entrypoint}.meta.json"
        target_meta = Path(str(target) + ".meta.json")
        if source_meta.exists() and source_meta.is_file():
            shutil.copyfile(source_meta, target_meta)

        return {
            "package_id": package_id,
            "installed_model_path": str(target),
            "installed_model_sha256": _sha256_file(target),
            "installed_meta_path": str(target_meta) if target_meta.exists() else None,
        }

    def show_package(self, package_id: str) -> dict[str, Any]:
        return self._find_package_record(package_id)

    def remote_push_placeholder(self, package_id: str, server_url: str) -> dict[str, Any]:
        self._find_package_record(package_id)
        return {
            "implemented": False,
            "message": (
                "Remote push is not implemented yet. Local package manifests are ready for server sync later."
            ),
            "package_id": package_id,
            "server_url": server_url,
        }

    def remote_search_placeholder(self, server_url: str, query: str) -> dict[str, Any]:
        return {
            "implemented": False,
            "message": "Remote search is not implemented yet.",
            "server_url": server_url,
            "query": query,
        }


def _print(payload: Any, as_json: bool):
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if isinstance(payload, list):
        if not payload:
            print("(no packages)")
            return
        for row in payload:
            print(
                f"{row.get('package_id')} | {row.get('name')} {row.get('version')} | "
                f"author={row.get('author')} | tags={','.join(row.get('tags', []))}"
            )
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
        return
    print(payload)


def main():
    parser = argparse.ArgumentParser(description="Ickle Model Library (local-first package registry).")
    parser.add_argument("--root", default="data/model_library", help="Library root path")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List known model packages")

    show = sub.add_parser("show", help="Show package metadata")
    show.add_argument("--package-id", required=True)

    export = sub.add_parser("export", help="Export local model checkpoint as a library package")
    export.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    export.add_argument("--name", required=True, help="Human-readable model name")
    export.add_argument("--version", required=True, help="Version label (e.g. 1.0.0)")
    export.add_argument("--author", required=True, help="Author/owner label")
    export.add_argument("--summary", default="", help="Short summary")
    export.add_argument("--tag", action="append", default=[], help="Optional tag (repeatable)")
    export.add_argument("--overwrite", action="store_true", help="Overwrite existing package id")

    imp = sub.add_parser("import", help="Import package directory into local library")
    imp.add_argument("--package-dir", required=True, help="Path containing manifest.json")
    imp.add_argument("--overwrite", action="store_true", help="Overwrite package if it exists")

    validate = sub.add_parser("validate", help="Validate package directory manifest/files")
    validate.add_argument("--package-dir", required=True)

    install = sub.add_parser("install", help="Install package model checkpoint into models/")
    install.add_argument("--package-id", required=True)
    install.add_argument("--out", default="", help="Optional output model path")
    install.add_argument("--overwrite", action="store_true", help="Overwrite output model if it exists")

    remote_push = sub.add_parser("remote-push", help="Placeholder for future server upload")
    remote_push.add_argument("--package-id", required=True)
    remote_push.add_argument("--server-url", required=True)

    remote_search = sub.add_parser("remote-search", help="Placeholder for future server search")
    remote_search.add_argument("--server-url", required=True)
    remote_search.add_argument("--query", required=True)

    args = parser.parse_args()
    library = ModelLibrary(root=args.root)

    try:
        if args.command == "list":
            _print(library.list_packages(), args.json)
            return
        if args.command == "show":
            _print(library.show_package(args.package_id), args.json)
            return
        if args.command == "export":
            out = library.export_package(
                model_path=args.model,
                name=args.name,
                version=args.version,
                author=args.author,
                summary=args.summary,
                tags=list(args.tag),
                overwrite=bool(args.overwrite),
            )
            _print(out, args.json)
            return
        if args.command == "import":
            out = library.import_package(args.package_dir, overwrite=bool(args.overwrite))
            _print(out, args.json)
            return
        if args.command == "validate":
            out = library.validate_package(args.package_dir)
            _print(out, args.json)
            return
        if args.command == "install":
            out = library.install_package(args.package_id, out_path=args.out or None, overwrite=bool(args.overwrite))
            _print(out, args.json)
            return
        if args.command == "remote-push":
            out = library.remote_push_placeholder(args.package_id, args.server_url)
            _print(out, args.json)
            return
        if args.command == "remote-search":
            out = library.remote_search_placeholder(args.server_url, args.query)
            _print(out, args.json)
            return
    except ModelLibraryError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"error: {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

