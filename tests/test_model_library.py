import hashlib
import json
import unittest
import uuid
from pathlib import Path

from src.model_library import ModelLibrary


class ModelLibraryTests(unittest.TestCase):
    @staticmethod
    def _tmp_dir(name: str) -> Path:
        root = Path("data") / ".tmp_tests" / f"{name}_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_export_and_install_roundtrip(self):
        workspace = self._tmp_dir("model_library")
        models = workspace / "models"
        models.mkdir(parents=True, exist_ok=True)
        source_model = models / "sample.pt"
        source_model.write_bytes(b"fake-model-bytes")
        source_meta = Path(str(source_model) + ".meta.json")
        source_meta.write_text(json.dumps({"steps": 10}), encoding="utf-8")

        library_root = workspace / "library"
        lib = ModelLibrary(root=str(library_root))
        exported = lib.export_package(
            model_path=str(source_model),
            name="Mechanic Expert",
            version="1.0.0",
            author="Local User",
            summary="Mechanical troubleshooting model",
            tags=["mechanic", "automotive"],
        )
        package_id = exported["package_id"]
        listed = lib.list_packages()
        self.assertTrue(any(row.get("package_id") == package_id for row in listed))

        install_path = workspace / "installed" / "mechanic.pt"
        installed = lib.install_package(package_id, out_path=str(install_path))
        self.assertTrue(install_path.exists())
        self.assertEqual(install_path.read_bytes(), source_model.read_bytes())
        self.assertEqual(installed["package_id"], package_id)

        validation = lib.validate_package(exported["package_dir"])
        self.assertTrue(validation["valid"])

    def test_import_external_package(self):
        workspace = self._tmp_dir("model_library_import")
        external = workspace / "external_pkg"
        external.mkdir(parents=True, exist_ok=True)
        model_file = external / "expert.pt"
        model_file.write_bytes(b"another-model")
        sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
        manifest = {
            "schema_version": "1.0",
            "package_id": "expert-pack-1",
            "name": "Expert Pack",
            "version": "1.0",
            "author": "User",
            "summary": "Imported package",
            "tags": ["expert"],
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "entrypoint": "expert.pt",
            "files": [
                {
                    "path": "expert.pt",
                    "size_bytes": len(model_file.read_bytes()),
                    "sha256": sha,
                }
            ],
        }
        (external / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        library_root = workspace / "library"
        lib = ModelLibrary(root=str(library_root))
        imported = lib.import_package(str(external))
        self.assertEqual(imported["package_id"], "expert-pack-1")
        shown = lib.show_package("expert-pack-1")
        self.assertEqual(shown["name"], "Expert Pack")


if __name__ == "__main__":
    unittest.main()
