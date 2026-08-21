import sys
import unittest
from pathlib import Path

from src.workspace_paths import WorkspaceLayout, get_app_root, get_project_root


class WorkspacePathTests(unittest.TestCase):
    def test_detect_simple_separation(self):
        layout = WorkspaceLayout(
            project_root=Path("C:/Projects/Ickle"),
            training_root=Path("C:/Projects/IckleTraining"),
        )
        self.assertTrue(layout.separated)
        self.assertFalse(layout.training_inside_project)
        self.assertFalse(layout.project_inside_training)

    def test_detect_nested_training(self):
        layout = WorkspaceLayout(
            project_root=Path("C:/Projects/Ickle"),
            training_root=Path("C:/Projects/Ickle/data/training"),
        )
        self.assertTrue(layout.separated)
        self.assertTrue(layout.training_inside_project)


class AppRootTests(unittest.TestCase):
    def test_get_app_root_is_project_root_when_not_frozen(self):
        self.assertFalse(hasattr(sys, "_MEIPASS"))
        self.assertEqual(get_app_root(), get_project_root())

    def test_get_app_root_uses_meipass_when_frozen(self):
        sys._MEIPASS = "C:/fake/frozen/_internal"  # type: ignore[attr-defined]
        try:
            self.assertEqual(get_app_root(), Path("C:/fake/frozen/_internal"))
        finally:
            del sys._MEIPASS  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()

