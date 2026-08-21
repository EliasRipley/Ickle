import os
import tempfile
import unittest
from unittest import mock

from src.tools.notepad_writer import write_in_notepad


class NotepadWriterPathTraversalTests(unittest.TestCase):
    """Regression coverage for a real path-traversal bug: write_in_notepad
    used os.path.join(temp_dir, filename) directly. os.path.join silently
    discards temp_dir when filename is absolute (e.g. "C:\\Windows\\x.txt"),
    and "..\\..\\x.txt" walks out of temp_dir even as a relative path --
    either way, a malicious/compromised tool call could write an arbitrary
    file anywhere the process has permission to write."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        patcher = mock.patch("src.tools.notepad_writer.tempfile.gettempdir", return_value=self._tmpdir)
        patcher.start()
        self.addCleanup(patcher.stop)
        startfile_patcher = mock.patch("os.startfile", create=True)
        startfile_patcher.start()
        self.addCleanup(startfile_patcher.stop)

    def test_absolute_path_filename_is_confined_to_temp_dir(self):
        malicious = os.path.join(tempfile.gettempdir(), "..", "escaped_absolute.txt")
        path = write_in_notepad("hello", filename=malicious)
        self.assertEqual(os.path.dirname(path), os.path.normpath(self._tmpdir))
        self.assertTrue(os.path.isfile(path))

    def test_relative_traversal_filename_is_confined_to_temp_dir(self):
        path = write_in_notepad("hello", filename="..\\..\\escaped_relative.txt")
        self.assertEqual(os.path.dirname(path), os.path.normpath(self._tmpdir))
        self.assertTrue(os.path.isfile(path))

    def test_normal_filename_still_works(self):
        path = write_in_notepad("hello world", filename="note.txt")
        self.assertEqual(path, os.path.normpath(os.path.join(self._tmpdir, "note.txt")))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "hello world")


if __name__ == "__main__":
    unittest.main()
