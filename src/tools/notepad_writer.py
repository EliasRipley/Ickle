import os
import subprocess
import tempfile


def write_in_notepad(text: str, filename: str = "agent_note.txt") -> str:
    """Create a text file and open it in Windows Notepad."""
    temp_dir = tempfile.gettempdir()

    # filename is agent/tool-callable input. os.path.join silently discards
    # temp_dir if filename is an absolute path (e.g. "C:\\Windows\\..."), and
    # "..\\..\\x.txt" traverses out of temp_dir even though it's relative --
    # both let a malicious tool call write anywhere the process can reach.
    # Strip to a bare filename and re-verify the resolved path is still
    # inside temp_dir before ever opening it for write.
    safe_name = os.path.basename(str(filename).strip()) or "agent_note.txt"
    path = os.path.normpath(os.path.join(temp_dir, safe_name))
    if os.path.commonpath([os.path.normpath(temp_dir), path]) != os.path.normpath(temp_dir):
        raise ValueError(f"Refusing to write outside the temp directory: {filename!r}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    os.startfile(path)
    return path
