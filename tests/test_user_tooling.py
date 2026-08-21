import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.user_tooling import ToolPermissionError, UserToolRegistry

TMP_ROOT = Path("data/.tmp_tooling")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


class UserToolingTests(unittest.TestCase):
    @contextmanager
    def _tool_dir(self):
        tools = TMP_ROOT / f"tooling-{uuid.uuid4().hex}"
        tools.mkdir(parents=True, exist_ok=False)
        try:
            yield tools
        finally:
            shutil.rmtree(tools, ignore_errors=True)

    def test_load_and_run_user_tool(self):
        with self._tool_dir() as tools:
            tool_file = tools / "echo.py"
            tool_file.write_text(
                "def run(payload):\n"
                "    return f\"ECHO:{payload.get('msg','')}\"\n",
                encoding="utf-8",
            )

            reg = UserToolRegistry(str(tools))
            self.assertEqual(reg.list_tools(), ["echo"])
            out = reg.run_tool("echo", '{"msg":"hi"}')
            self.assertEqual(out, "ECHO:hi")

    def test_permission_blocks_network_import_by_default(self):
        with self._tool_dir() as tools:
            tool_file = tools / "net_tool.py"
            tool_file.write_text(
                "import urllib.request\n"
                "def run(payload):\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            reg = UserToolRegistry(str(tools))
            with self.assertRaises(ToolPermissionError):
                reg.run_tool("net_tool", "{}")

    def test_docstring_mentioning_import_requests_is_not_a_false_positive(self):
        # Regression: the old check was a raw substring scan for "import
        # requests" over the whole file text, so a docstring or comment
        # that merely *mentions* an import would incorrectly block a tool
        # that imports nothing risky at all.
        with self._tool_dir() as tools:
            tool_file = tools / "safe_tool.py"
            tool_file.write_text(
                '"""Example: do NOT `import requests` in this tool."""\n'
                "def run(payload):\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            reg = UserToolRegistry(str(tools))
            self.assertEqual(reg.run_tool("safe_tool", "{}"), "ok")

    def test_whitespace_obfuscated_import_is_still_blocked(self):
        # Regression: "import  requests" (double space) evaded the old
        # substring check ("import requests" != "import  requests") but is
        # a perfectly normal, real import as far as Python is concerned.
        with self._tool_dir() as tools:
            tool_file = tools / "sneaky_net.py"
            tool_file.write_text(
                "import  requests\n"
                "def run(payload):\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            reg = UserToolRegistry(str(tools))
            with self.assertRaises(ToolPermissionError):
                reg.run_tool("sneaky_net", "{}")

    def test_dynamic_import_of_subprocess_is_blocked(self):
        # Regression: __import__("subprocess") never matched the old
        # "import subprocess" substring check at all.
        with self._tool_dir() as tools:
            tool_file = tools / "dyn_proc.py"
            tool_file.write_text(
                "def run(payload):\n"
                "    sp = __import__('subprocess')\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            reg = UserToolRegistry(str(tools))
            with self.assertRaises(ToolPermissionError):
                reg.run_tool("dyn_proc", "{}")

    def test_permission_manifest_can_enable_network(self):
        with self._tool_dir() as tools:
            tool_file = tools / "net_ok.py"
            tool_file.write_text(
                "import urllib.request\n"
                "def run(payload):\n"
                "    return 'allowed'\n",
                encoding="utf-8",
            )
            (tools / "net_ok.tool.json").write_text(
                '{"permissions":{"network":true,"process":false,"filesystem_read":true,"filesystem_write":false}}',
                encoding="utf-8",
            )
            reg = UserToolRegistry(str(tools))
            out = reg.run_tool("net_ok", "{}")
            self.assertEqual(out, "allowed")


if __name__ == "__main__":
    unittest.main()
