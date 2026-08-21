"""Ickle as an MCP server, so IDEs and agent tools (Claude Code, Cursor, etc.)
can summon it directly instead of needing a browser tab or a separate CLI
invocation. This is the standard way local tools integrate with IDEs today --
Ickle has no npm ecosystem and gains nothing by inventing one.

Exposes a small, honest set of tools rather than everything the CLI can do:
chat (the actual product), check_capability (reuses the existing
capability-honesty system instead of inventing new claims about what Ickle
can do), training_status (read-only diagnostics), image_understanding
(OCR + captioning), and code tools (read_file, write_file, edit_file,
run_command, search_repo -- wrapping src/code_agent.py). Unlike the chat/
agent path, the code tools here are not gated behind an opt-in flag: the
calling IDE or agent client's own tool-permission system is the primary
trust boundary for MCP tools, the same way it already is for other
IDE-integrated coding agents. That said, relying on the client alone was a
real gap -- not every MCP client enforces per-call confirmation, and
src/code_agent.py's own functions do no path containment by design (its
CLI is meant to point anywhere via --workspace). So read_file/write_file/
edit_file/run_command here additionally confine paths (and run_command's
cwd) to the process's working directory via the same
_resolve_within_workspace() check src/agent_loop.py's chat/agent path
already uses, as defense-in-depth rather than a single point of trust. Run
with:

    python -m src.app mcp-server

Then point an MCP-compatible client at it as a stdio server, e.g. in a
`.mcp.json`:
    {"mcpServers": {"ickle": {"command": "python", "args": ["-m", "src.app", "mcp-server"]}}}
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from mcp.server.fastmcp import FastMCP

from src.agent_loop import _resolve_within_workspace
from src.capabilities import check_capability as _check_capability
from src.code_agent import edit_file as _edit_file
from src.code_agent import read_file as _read_file
from src.code_agent import run_command as _run_command
from src.code_agent import search_repo as _search_repo
from src.code_agent import write_file as _write_file
from src.ilm_chat import _resolve_default_model, generate_response
from src.tools.image_reader import ImageToolsUnavailable, describe_image, extract_text_from_image
from src.training_control import inspect_training_status
from src.workspace_paths import get_training_root

mcp = FastMCP(
    name="ickle",
    instructions="Ickle is a local-first, optionally-federated small language model. "
    "Use `chat` to talk to the locally trained model, `check_capability` to honestly "
    "check whether a task is supported before claiming it is, and `training_status` "
    "to check on an in-progress training run.",
)


@mcp.tool()
def chat(prompt: str, model: str = "") -> str:
    """Send a prompt to the locally trained Ickle model and return its response.

    `model` is an optional path to a specific checkpoint; if omitted, Ickle's
    normal default-model resolution is used (the most recently promoted model).
    """
    no_model_message = (
        "No trained model is available yet in this Ickle installation "
        "(models/ is empty, or the given `model` path doesn't exist). Train "
        "one first with `python -m src.app train`, or point `model` at an "
        "existing checkpoint."
    )
    try:
        resolved_model = model.strip() or _resolve_default_model()
    except FileNotFoundError:
        return no_model_message

    args = SimpleNamespace(
        model=resolved_model,
        prompt=prompt,
        max_new=260,
        max_new_limit=500,
        temperature=0.25,
        top_k=20,
        torch_threads=4,
        skill="",
        enable_memory=True,
        enable_web_tools=True,
        speculative=False,
        speculative_gamma=3,
        thinking_mode=False,
        thinking=False,
        think_budget=0,
        agent=False,
        agent_mode=False,
        autonomy_mode=None,
    )
    try:
        result = generate_response(args)
    except FileNotFoundError:
        # Reachable when an explicit `model` path is given but doesn't exist --
        # the default-resolution path above only catches the no-args case.
        return no_model_message
    return str(result.get("response", "")).strip()


@mcp.tool()
def image_understanding(image_path: str) -> str:
    """Read an image file: extract any visible text (OCR) and generate a
    short description of its contents. Ickle's own model stays text-only --
    this wraps optional vision libraries (requirements-vision.txt) the same
    way `chat`'s image attachment support does in the web app."""
    parts: list[str] = []
    try:
        text = extract_text_from_image(image_path)
        if text:
            parts.append(f"Text found in the image:\n{text}")
    except ImageToolsUnavailable as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"Could not read text from the image: {exc}"

    try:
        description = describe_image(image_path)
        if description:
            parts.append(f"Image description: {description}")
    except ImageToolsUnavailable as exc:
        parts.append(f"(Image description unavailable: {exc})")
    except Exception as exc:  # noqa: BLE001
        parts.append(f"(Error describing the image: {exc})")

    return "\n\n".join(parts) if parts else "No text or description could be extracted from this image."


@mcp.tool()
def read_file(file_path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file from the local workspace, optionally a specific line range
    (1-based, 0 means unbounded on that side). Read-only."""
    try:
        return _read_file(_resolve_within_workspace(file_path), start_line, end_line)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return f"Could not read {file_path}: {exc}"


@mcp.tool()
def search_repo(pattern: str, repo_root: str = "", file_types: str = "*") -> str:
    """Search the local workspace for a regex pattern. `file_types` is an
    optional comma-separated list of extensions (e.g. "py,js"), "*" for all.
    Read-only."""
    try:
        confined_root = _resolve_within_workspace(repo_root) if repo_root else ""
    except ValueError as exc:
        return str(exc)
    hits = _search_repo(pattern, repo_root=confined_root, file_types=file_types)
    if not hits:
        return "No matches found."
    return "\n".join(f"{h['file']}:{h['line']}: {h['context']}" for h in hits[:50])


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the local workspace, creating it or
    overwriting it entirely. Unlike the chat/agent path, this is not gated
    behind an opt-in flag -- the calling IDE or agent client's own
    tool-permission system is the trust boundary for MCP tools, the same way
    it already is for other IDE-integrated coding agents."""
    try:
        return _write_file(_resolve_within_workspace(file_path), content)
    except (OSError, ValueError) as exc:
        return f"Could not write {file_path}: {exc}"


@mcp.tool()
def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact string in a file with a new string. Fails if
    old_string isn't found, or (unless replace_all is set) if it matches more
    than once -- provide more surrounding context to disambiguate."""
    try:
        return _edit_file(_resolve_within_workspace(file_path), old_string, new_string, replace_all)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return f"Could not edit {file_path}: {exc}"


@mcp.tool()
def run_command(cmd: str, cwd: str = "", timeout_seconds: int = 120) -> str:
    """Run a shell command in the local workspace and return its exit code,
    stdout, and stderr as JSON. `cwd`, if given, must stay within the
    workspace -- the command text itself isn't sandboxed beyond that (this
    tool's whole purpose is running arbitrary commands), so treat granting
    an MCP client access to it the same as granting shell access."""
    try:
        confined_cwd = _resolve_within_workspace(cwd) if cwd else ""
        return _run_command(cmd, confined_cwd, timeout_seconds)
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return f"Could not run command: {exc}"


@mcp.tool()
def check_capability(task_text: str) -> str:
    """Honestly report whether Ickle currently supports a given task, instead
    of guessing or overclaiming. Reuses the same capability-honesty system the
    chat path and hub REPL already use (src/capabilities.py)."""
    report = _check_capability(task_text)
    if report.supported:
        return f"SUPPORTED: {report.summary}"
    if report.suggestion:
        return f"NOT SUPPORTED: {report.summary} Suggestion: {report.suggestion}"
    return f"NOT SUPPORTED: {report.summary}"


@mcp.tool()
def training_status() -> str:
    """Report whether a training run is currently active, and its latest
    progress numbers (step, loss, etc.) if so. Read-only -- does not start,
    stop, or otherwise control training."""
    status = inspect_training_status(get_training_root() / "training_live.json")
    if not status.get("exists"):
        return "No training run has been recorded yet."
    if status.get("is_stale"):
        return f"Last known training status is stale ({status.get('stale_reason', 'unknown reason')})."
    lines = [f"status: {status.get('reported_status', 'unknown')}"]
    for key in ("step", "total_steps", "train_loss", "val_loss", "perplexity", "acc_top1", "acc_top5", "lr"):
        if key in status:
            lines.append(f"{key}: {status[key]}")
    return "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
