import unittest

try:
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:
    HAS_MCP = False


@unittest.skipUnless(HAS_MCP, "mcp is an optional dependency (requirements-mcp.txt), not installed here")
class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    """Regression coverage for `python -m src.app mcp-server` -- the IDE/agent
    integration path. Ickle has no npm ecosystem; MCP over stdio is the
    standard way local tools integrate with IDEs today."""

    def _server(self):
        from src.mcp_server import mcp as server

        return server

    async def test_lists_expected_tools(self):
        server = self._server()
        tools = await server.list_tools()
        names = {t.name for t in tools}
        self.assertEqual(
            names,
            {
                "chat",
                "check_capability",
                "training_status",
                "image_understanding",
                "read_file",
                "write_file",
                "edit_file",
                "run_command",
                "search_repo",
            },
        )

    async def test_check_capability_tool_reuses_capability_honesty_system(self):
        server = self._server()
        _content, structured = await server.call_tool("check_capability", {"task_text": "set a timer"})
        self.assertIn("SUPPORTED", structured["result"])

    async def test_chat_tool_reports_missing_model_honestly_instead_of_crashing(self):
        server = self._server()
        # A prompt unlikely to be short-circuited by local_reasoning (greetings/
        # arithmetic), so this actually reaches _load_model_bundle() and
        # exercises the bad-model-path failure handling in chat().
        _content, structured = await server.call_tool(
            "chat",
            {"prompt": "Explain gradient checkpointing in detail please", "model": "models/does_not_exist.pt"},
        )
        self.assertIn("No trained model is available", structured["result"])

    async def test_training_status_tool_reports_no_run_recorded(self):
        server = self._server()
        _content, structured = await server.call_tool(
            "training_status", {}
        )
        self.assertIsInstance(structured["result"], str)


if __name__ == "__main__":
    unittest.main()
