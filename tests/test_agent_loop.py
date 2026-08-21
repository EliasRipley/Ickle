import unittest
from pathlib import Path
from types import SimpleNamespace

from src.agent_loop import AgentResult, ToolDef, _code_tools, _resolve_within_workspace, agent_loop
from src.ilm_chat_generation import generate_reasoning_text
from src.ilm_chat import _format_agent_trace
from src.model import ILM, TinyConfig
from src.system_limits import SystemLimits
from src.tokenizer import CharTokenizer


def _tiny_model_and_tokenizer():
    tokenizer = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,?<>reason")
    cfg = TinyConfig(vocab_size=tokenizer.vocab_size, block_size=32, n_embd=16, n_head=2, n_layer=1)
    model = ILM(cfg)
    model.eval()
    return model, tokenizer


class GenerateReasoningTextTests(unittest.TestCase):
    """Regression test for a real crash: generate_streaming() yields a [1,1]
    tensor per step (batch=1, one new token), not a scalar.
    torch.cat(reason_ids).tolist() on a list of those produces nested lists
    ([[id], [id], ...]), which tokenizer.decode() can't int() over --
    reproduced live via agent_mode + thinking_mode together before this fix.
    Uses a real tiny ILM model (not a mock) so the test actually exercises
    the real tensor shape that caused the bug."""

    def test_generate_reasoning_text_does_not_crash_on_real_model_output(self):
        model, tokenizer = _tiny_model_and_tokenizer()
        reasoning, token_count = generate_reasoning_text(
            model, tokenizer, "User: hello\n\n<reason>", max_tokens=5, temperature=0.8, top_k=10
        )
        self.assertIsInstance(reasoning, str)
        self.assertEqual(token_count, 5)


class AgentLoopReasoningTests(unittest.TestCase):
    """agent_loop's reasoning_enabled branch previously carried its own,
    independently-drifted copy of the same token-decoding logic
    generate_reasoning_text now provides -- this exercises that branch
    end to end (not just the extracted helper) with a real tiny model."""

    def test_agent_loop_with_thinking_enabled_does_not_crash(self):
        model, tokenizer = _tiny_model_and_tokenizer()
        limits = SystemLimits(max_new_tokens=20)
        args = SimpleNamespace(max_new=20, top_k=10, temperature=0.8, thinking_mode=True)

        result = agent_loop(model, tokenizer, "System prompt", "hello", args, limits, tools=[])
        self.assertIsInstance(result, AgentResult)
        self.assertIsInstance(result.response, str)


class FormatAgentTraceTests(unittest.TestCase):
    def test_includes_reasoning_and_each_tool_call_with_result_or_error(self):
        result = AgentResult(
            response="The answer is 42.",
            reasoning="I should look this up.",
            steps=2,
            tool_calls=[
                {"tool": "web_read", "params": {"url": "http://example.com"}, "result": "Example page content."},
                {"tool": "memory_search", "params": {"query": "favorite number"}, "error": "No results found."},
            ],
        )
        trace = _format_agent_trace(result)
        self.assertIn("I should look this up.", trace)
        self.assertIn("Step 1: called web_read(url=http://example.com)", trace)
        self.assertIn("Example page content.", trace)
        self.assertIn("Step 2: called memory_search(query=favorite number)", trace)
        self.assertIn("error: No results found.", trace)

    def test_empty_tool_calls_still_includes_reasoning(self):
        result = AgentResult(response="Hi.", reasoning="Just a greeting.", steps=1, tool_calls=[])
        trace = _format_agent_trace(result)
        self.assertEqual(trace, "Just a greeting.")


class AgentLoopToolCallRecordTests(unittest.TestCase):
    """agent_loop used to only record {"tool", "params"} per call, discarding
    the actual result/error -- meaning even a correctly-surfaced trace would
    show what was attempted but never what happened. Verifies the recorded
    call now carries the outcome, using a stubbed model response (a real
    untrained tiny model won't reliably emit well-formed <call> XML)."""

    def test_successful_tool_call_records_result(self):
        from unittest import mock

        model, tokenizer = _tiny_model_and_tokenizer()
        limits = SystemLimits(max_new_tokens=20)
        args = SimpleNamespace(max_new=20, top_k=10, thinking_mode=False)
        tools = [ToolDef("echo", "Echoes text back", [("text", "text to echo")], lambda text: f"echoed:{text}")]

        responses = iter([
            "<call><tool>echo</tool><text>hi</text></call>",
            "All done.",
        ])
        with mock.patch("src.agent_loop._generate_model_response", side_effect=lambda *a, **k: next(responses)):
            result = agent_loop(model, tokenizer, "System prompt", "say hi", args, limits, tools=tools)

        self.assertEqual(result.response, "All done.")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["tool"], "echo")
        self.assertEqual(result.tool_calls[0]["result"], "echoed:hi")
        self.assertNotIn("error", result.tool_calls[0])

    def test_failing_tool_call_records_error(self):
        from unittest import mock

        model, tokenizer = _tiny_model_and_tokenizer()
        limits = SystemLimits(max_new_tokens=20)
        args = SimpleNamespace(max_new=20, top_k=10, thinking_mode=False)

        def _boom(text):
            raise RuntimeError("tool exploded")

        tools = [ToolDef("boom", "Always fails", [("text", "text")], _boom)]

        responses = iter([
            "<call><tool>boom</tool><text>hi</text></call>",
            "Done despite the error.",
        ])
        with mock.patch("src.agent_loop._generate_model_response", side_effect=lambda *a, **k: next(responses)):
            result = agent_loop(model, tokenizer, "System prompt", "trigger boom", args, limits, tools=tools)

        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn("tool exploded", result.tool_calls[0]["error"])
        self.assertNotIn("result", result.tool_calls[0])


class CodeToolsGatingTests(unittest.TestCase):
    def test_read_only_tools_available_without_execution_flag(self):
        names = {t.name for t in _code_tools(False)}
        self.assertEqual(names, {"read_file", "search_repo"})

    def test_write_tools_only_available_with_execution_flag(self):
        names = {t.name for t in _code_tools(True)}
        self.assertEqual(names, {"read_file", "search_repo", "write_file", "edit_file", "run_command"})


class ResolveWithinWorkspaceTests(unittest.TestCase):
    """Regression coverage for a real gap: src/code_agent.py itself does no
    path containment (by design -- its CLI is meant to point at any
    directory), so without a check here a chat request (a much lower-trust
    caller than someone running the CLI directly) could read or write files
    anywhere the process has permission for via `../../` traversal or a bare
    absolute path."""

    def test_relative_path_inside_workspace_resolves(self):
        resolved = _resolve_within_workspace("src/agent_loop.py")
        self.assertTrue(Path(resolved).exists())

    def test_parent_traversal_outside_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_within_workspace("../../../../Windows/System32/drivers/etc/hosts")

    def test_absolute_path_outside_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_within_workspace("C:/Windows/System32/drivers/etc/hosts")


class CodeToolWrapperRoundTripTests(unittest.TestCase):
    """Exercises the actual wrapper closures agent_loop hands to the model
    -- string-typed XML params in, real file I/O out -- against real files,
    not just the underlying src/code_agent.py functions directly."""

    def setUp(self):
        from uuid import uuid4

        self.tools = {t.name: t for t in _code_tools(True)}
        self.rel_dir = Path("data/.tmp_tests") / f"agent_code_tools_{uuid4().hex}"
        self.rel_dir.mkdir(parents=True, exist_ok=True)
        self.rel_path = self.rel_dir / "sample.txt"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.rel_dir, ignore_errors=True)

    def test_write_read_edit_round_trip(self):
        import json

        write_result = json.loads(self.tools["write_file"].fn(file_path=str(self.rel_path), content="hello agent"))
        self.assertEqual(write_result["size"], len("hello agent"))
        self.assertTrue(self.rel_path.exists())

        read_result = self.tools["read_file"].fn(file_path=str(self.rel_path))
        self.assertEqual(read_result, "hello agent")

        edit_result = self.tools["edit_file"].fn(
            file_path=str(self.rel_path), old_string="hello", new_string="goodbye", replace_all="false"
        )
        self.assertIn("goodbye", self.rel_path.read_text(encoding="utf-8"))
        self.assertTrue(edit_result)

    def test_write_file_outside_workspace_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tools["write_file"].fn(file_path="../../outside_workspace.txt", content="should not land")

    def test_run_command_executes_and_returns_output(self):
        result = self.tools["run_command"].fn(cmd="echo agent-tool-smoke-test", cwd="", timeout_seconds="30")
        self.assertIn("agent-tool-smoke-test", result)

    def test_search_repo_finds_known_symbol(self):
        result = self.tools["search_repo"].fn(pattern="_resolve_within_workspace", file_types="py")
        self.assertIn("agent_loop.py", result)


if __name__ == "__main__":
    unittest.main()
