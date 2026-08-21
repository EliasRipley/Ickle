"""Agent loop â€” SLM as orchestrator with D-CoT structured tool calling.

Uses the Agent-as-Tool paradigm (arXiv 2604.17009, 2026):
small model plans and dispatches, tool functions execute.
Tools are abstracted into a standardized XML action space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.icklization import ick
from src.ilm_chat_generation import GenerationResult, _generate_model_response, generate_reasoning_text
from src.system_limits import SystemLimits, clamp_new_tokens


MAX_STEPS = 15
MAX_CONTEXT_TOKENS = 2048


@dataclass
class ToolDef:
    name: str
    description: str
    params: list[tuple[str, str]]
    fn: Callable[..., str]


@dataclass
class AgentResult:
    response: str
    reasoning: str = ""
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


TOOL_CALL_RE = re.compile(
    r'<call>\s*<tool>(\w+)</tool>\s*(.*?)</call>', re.DOTALL
)
PARAM_RE = re.compile(r'<(\w+)>(.*?)</\w+>', re.DOTALL)

TOOL_DESCRIPTION_TEMPLATE = (
    '<call>\n'
    '  <tool>{name}</tool>\n'
    '  <{pname}>{pdesc}</{pname}>\n'
    '</call>'
)


def _make_tool_descriptions(tools: list[ToolDef]) -> str:
    lines = []
    for t in tools:
        params_str = "\n  ".join(
            f"<{pn}>: {pd}" for pn, pd in t.params
        )
        lines.append(f"- {t.name}: {t.description}\n  {params_str}")
    return "\n".join(lines)


def _parse_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        params = {}
        for p in PARAM_RE.finditer(body):
            params[p.group(1)] = p.group(2).strip()
        calls.append((name, params))
    return calls


def _trim_context(conversation: list[str], tokenizer, max_tokens: int):
    while len(tokenizer.encode("\n\n".join(conversation))) > max_tokens:
        if len(conversation) <= 3:
            break
        conversation.pop(2)


def agent_loop(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    args,
    limits: SystemLimits,
    tools: list[ToolDef] | None = None,
) -> AgentResult:
    if tools is None:
        tools = _default_tools() + _code_tools(bool(getattr(args, "allow_code_execution", False)))

    tool_registry = {t.name: t for t in tools}
    tool_descriptions = _make_tool_descriptions(tools)

    reasoning_enabled = getattr(args, "thinking_mode", False)
    all_reasoning: list[str] = []
    all_tool_calls: list[dict[str, Any]] = []
    conversation = [
        f"{system_prompt}\n\nAvailable tools:\n{tool_descriptions}",
        f"User: {user_prompt}",
    ]
    steps = 0

    while steps < MAX_STEPS:
        steps += 1
        prompt = "\n\n".join(conversation) + f"\n\n{ick.assistant_label()}:"

        if reasoning_enabled:
            reasoning_prompt = prompt + "\n\n<reason>"
            reason_tokens = min(100, clamp_new_tokens(args.max_new, limits.max_new_tokens))
            reasoning_clean, _ = generate_reasoning_text(
                model, tokenizer, reasoning_prompt, max_tokens=reason_tokens, temperature=0.4, top_k=args.top_k
            )
            if reasoning_clean:
                all_reasoning.append(reasoning_clean)
                prompt = prompt + f"\n\n<reason>{reasoning_clean}</reason>"

        response = _generate_model_response(model, tokenizer, prompt, args, limits)

        tool_calls = _parse_tool_calls(response)
        if not tool_calls:
            clean = TOOL_CALL_RE.sub("", response).strip()
            return AgentResult(
                response=clean or response,
                reasoning="\n".join(all_reasoning),
                steps=steps,
                tool_calls=all_tool_calls,
            )

        clean_response = TOOL_CALL_RE.sub("", response).strip()
        if clean_response:
            conversation.append(f"{ick.assistant_label()}: {clean_response}")

        for name, params in tool_calls:
            call_record: dict[str, Any] = {"tool": name, "params": dict(params)}
            all_tool_calls.append(call_record)
            if name not in tool_registry:
                call_record["error"] = f"unknown tool '{name}'"
                conversation.append(f"Tool result: unknown tool '{name}'")
                continue
            try:
                tool = tool_registry[name]
                result = tool.fn(**params)
                result_str = str(result)[:1200]
                call_record["result"] = result_str
                conversation.append(f"Tool result ({name}):\n{result_str}")
            except Exception as e:
                call_record["error"] = str(e)
                conversation.append(f"Tool error ({name}): {e}")

        _trim_context(conversation, tokenizer, MAX_CONTEXT_TOKENS)

    return AgentResult(
        response="I ran out of steps. Here is what I have so far.",
        reasoning="\n".join(all_reasoning),
        steps=steps,
        tool_calls=all_tool_calls,
    )


def _default_tools() -> list[ToolDef]:
    from src.tools.firefox_reader import read_url_text

    def web_read(url: str) -> str:
        return read_url_text(url, timeout_ms=25000, max_chars=8000)

    from src.tools.news_research import search_news, format_news_digest

    def news_search(query: str) -> str:
        items = search_news(query, max_results=5)
        return format_news_digest(items)

    from src.ilm_memory import get_memory

    def memory_search(query: str) -> str:
        facts = get_memory().search_facts(query, limit=3)
        if not facts:
            return "No memory results found."
        return "\n".join(f"- {f.get('fact', f)}" for f in facts)

    def think(thought: str) -> str:
        return f"(reasoned: {thought[:500]})"

    return [
        ToolDef("web_read", "Read a webpage and extract text content", [("url", "The URL to read")], web_read),
        ToolDef("news_search", "Search recent news headlines", [("query", "Search topic")], news_search),
        ToolDef("memory_search", "Search saved memory facts", [("query", "Search query")], memory_search),
        ToolDef("think", "Reason step by step internally", [("thought", "Your reasoning")], think),
    ]


def _resolve_within_workspace(path: str) -> str:
    """Resolve `path` and refuse it if it escapes the current workspace root.

    src/code_agent.py itself does no containment checking -- by design, its
    CLI (`code-agent --workspace <dir>`) is a general-purpose tool meant to
    point at any directory. But here, these wrappers are the chat/agent
    path's trust boundary: a chat request is a far lower-trust caller than
    someone invoking the CLI directly, so a prompt-injected "read
    ../../../.ssh/id_rsa" must not be able to escape the project root. (The
    MCP tools in src/mcp_server.py deliberately don't add this check -- the
    IDE's own tool-permission system is that surface's trust boundary
    instead, per the project's stated design.)
    """
    root = Path.cwd().resolve()
    candidate = Path(path)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"'{path}' is outside the project workspace ({root}); refusing to access it.") from None
    return str(candidate)


def _code_tools(allow_execution: bool) -> list[ToolDef]:
    """read_file/search_repo wrap src/code_agent.py's existing implementations
    and are always available -- read-only, same trust level as web_read.
    write_file/edit_file/run_command wrap the same module's write-capable
    functions but are only included when the caller has explicitly opted in
    via args.allow_code_execution (the "Allow code execution" chat toggle),
    since letting a chat request write files or run shell commands on this
    device needs to be opt-in, not a default agent-mode capability."""
    from src.code_agent import edit_file, read_file, run_command, search_repo, write_file

    def read_file_tool(file_path: str, start_line: str = "0", end_line: str = "0") -> str:
        return read_file(_resolve_within_workspace(file_path), int(start_line or 0), int(end_line or 0))

    def search_repo_tool(pattern: str, file_types: str = "*") -> str:
        hits = search_repo(pattern, file_types=file_types or "*")
        if not hits:
            return "No matches found."
        return "\n".join(f"{h['file']}:{h['line']}: {h['context']}" for h in hits[:20])

    tools = [
        ToolDef(
            "read_file",
            "Read a file from the local project workspace, optionally a specific line range",
            [
                ("file_path", "Path to the file, relative to the project root"),
                ("start_line", "Optional 1-based start line (0 for the beginning)"),
                ("end_line", "Optional 1-based end line (0 for the end)"),
            ],
            read_file_tool,
        ),
        ToolDef(
            "search_repo",
            "Search the local project workspace for a regex pattern",
            [
                ("pattern", "Regex pattern to search for"),
                ("file_types", "Optional comma-separated extensions to filter by, e.g. py,js (default: all)"),
            ],
            search_repo_tool,
        ),
    ]

    if not allow_execution:
        return tools

    def write_file_tool(file_path: str, content: str) -> str:
        return write_file(_resolve_within_workspace(file_path), content)

    def edit_file_tool(file_path: str, old_string: str, new_string: str, replace_all: str = "false") -> str:
        replace_all_bool = str(replace_all).strip().lower() in {"1", "true", "yes"}
        return edit_file(_resolve_within_workspace(file_path), old_string, new_string, replace_all_bool)

    def run_command_tool(cmd: str, cwd: str = "", timeout_seconds: str = "120") -> str:
        resolved_cwd = _resolve_within_workspace(cwd) if cwd else ""
        return run_command(cmd, resolved_cwd, int(timeout_seconds or 120))

    tools += [
        ToolDef(
            "write_file",
            "Write content to a file in the local project workspace, creating or overwriting it",
            [("file_path", "Path to the file"), ("content", "The full new file content")],
            write_file_tool,
        ),
        ToolDef(
            "edit_file",
            "Replace an exact string in a file with a new string",
            [
                ("file_path", "Path to the file"),
                ("old_string", "The exact existing text to replace"),
                ("new_string", "The replacement text"),
                ("replace_all", "true/false -- replace every occurrence instead of requiring a unique match (default false)"),
            ],
            edit_file_tool,
        ),
        ToolDef(
            "run_command",
            "Run a shell command in the local project workspace and return its output",
            [
                ("cmd", "The shell command to run"),
                ("cwd", "Optional working directory (default: project root)"),
                ("timeout_seconds", "Optional timeout in seconds (default 120)"),
            ],
            run_command_tool,
        ),
    ]
    return tools
