from dataclasses import dataclass, field

from src.autonomy import get_mode, policy_summary
from src.capabilities import check_capability
from src.cloud_assist import assist as cloud_assist_call, cloud_status_text
from src.clarify import (
    ClarificationResult,
    detect_vague_prompt,
    minecraft_brief_clarification,
    social_brief_clarification,
)
from src.icklization import ick
from src.system_limits import SystemLimits
from src.tools.desktop_control import ALLOWED_ACTIONS as _DESKTOP_ALLOWED_ACTIONS, DISPATCH as _DESKTOP_DISPATCH
from src.tools.firefox_reader import read_url_text
from src.tools.minecraft_guide import fetch_minecraft_topic, list_beginner_topics
from src.tools.news_research import format_news_digest, search_news
from src.tools.notepad_writer import write_in_notepad
from src.tools.timer_tool import TimerManager, format_timer_status, parse_duration
from src.user_tooling import UserToolRegistry


@dataclass
class LocalAgent:
    limits: SystemLimits = field(default_factory=SystemLimits)
    tool_calls_used: int = 0
    autonomy_mode: str = "balanced"
    user_tools: UserToolRegistry = field(default_factory=UserToolRegistry)

    def _consume_tool_call(self):
        if self.tool_calls_used >= self.limits.max_tool_calls:
            raise RuntimeError("Tool-call budget exceeded.")
        self.tool_calls_used += 1


    def set_autonomy_mode(self, mode_name: str):
        mode = get_mode(mode_name)
        self.autonomy_mode = mode.name

    def get_policy_summary(self) -> str:
        return policy_summary(self.autonomy_mode)

    def capability_check(self, task_text: str) -> str:
        report = check_capability(task_text)
        if report.supported:
            return f"SUPPORTED: {report.summary}"
        if report.suggestion:
            return f"NOT SUPPORTED: {report.summary} Suggestion: {report.suggestion}"
        return f"NOT SUPPORTED: {report.summary}"

    def maybe_request_clarification(self, prompt: str) -> ClarificationResult:
        if detect_vague_prompt(prompt):
            return ClarificationResult(
                needs_clarification=True,
                question=ick.vague_prompt_question(),
            )
        return ClarificationResult(needs_clarification=False)

    def social_marketing_clarification(
        self,
        platform: str | None = None,
        audience: str | None = None,
        goal: str | None = None,
        offer: str | None = None,
    ) -> ClarificationResult:
        return social_brief_clarification(platform=platform, audience=audience, goal=goal, offer=offer)

    def minecraft_learning_clarification(
        self,
        edition: str | None = None,
        platform: str | None = None,
        experience: str | None = None,
        goal: str | None = None,
    ) -> ClarificationResult:
        return minecraft_brief_clarification(
            edition=edition,
            platform=platform,
            experience=experience,
            goal=goal,
        )

    def read_webpage(self, url: str) -> str:
        self._consume_tool_call()
        return read_url_text(
            url,
            timeout_ms=self.limits.web_timeout_ms,
            max_chars=self.limits.max_web_chars,
        )

    def research_marketing_topic(self, query: str, max_results: int = 5) -> str:
        self._consume_tool_call()
        items = search_news(query=query, max_results=max_results)
        return format_news_digest(items)

    def list_minecraft_topics(self) -> list[str]:
        return list_beginner_topics()

    def read_minecraft_topic(self, topic_key: str) -> str:
        self._consume_tool_call()
        return fetch_minecraft_topic(
            topic_key,
            timeout_ms=self.limits.web_timeout_ms,
            max_chars=self.limits.max_web_chars,
        )

    def cloud_status(self) -> str:
        return cloud_status_text()

    def cloud_assist(self, prompt: str, model: str | None = None) -> str:
        return cloud_assist_call(prompt=prompt, model=model)

    def list_user_tools(self) -> list[str]:
        return self.user_tools.list_tools()

    def run_user_tool(self, name: str, payload_json: str = "{}") -> str:
        self._consume_tool_call()
        return self.user_tools.run_tool(name=name, payload_json=payload_json)

    def write_note(self, text: str, filename: str = "agent_note.txt") -> str:
        self._consume_tool_call()
        return write_in_notepad(text, filename=filename)

    def timer_set(self, duration_text: str, name: str = "") -> str:
        self._consume_tool_call()
        seconds = parse_duration(duration_text)
        if seconds is None:
            return f"Could not parse duration: {duration_text}"
        tm = TimerManager.get_instance()
        timer = tm.set_timer(name=name or "timer", duration_seconds=seconds)
        return f"Timer set: {format_timer_status(timer)}"

    def timer_list(self) -> str:
        tm = TimerManager.get_instance()
        timers = tm.list_timers()
        if not timers:
            return "No timers active."
        return "\n".join(format_timer_status(t) for t in timers)

    def timer_check(self, name: str) -> str:
        tm = TimerManager.get_instance()
        timer = tm.get_timer(name)
        if timer is None:
            return f"No timer named '{name}'."
        return format_timer_status(timer)

    def timer_pause(self, name: str) -> str:
        tm = TimerManager.get_instance()
        timer = tm.pause_timer(name)
        if timer is None:
            return f"Could not pause timer '{name}'."
        return format_timer_status(timer)

    def timer_resume(self, name: str) -> str:
        tm = TimerManager.get_instance()
        timer = tm.resume_timer(name)
        if timer is None:
            return f"Could not resume timer '{name}'."
        return format_timer_status(timer)

    def timer_cancel(self, name: str) -> str:
        tm = TimerManager.get_instance()
        timer = tm.cancel_timer(name)
        if timer is None:
            return f"Could not cancel timer '{name}'."
        return f"Timer cancelled: {timer.name}"

    # Aliases the model/UI may send for an action that DISPATCH knows under
    # a different canonical name.
    _DESKTOP_ACTION_ALIASES = {
        "type": "type_text",
        "press": "key_press",
        "send": "desktop_send",
    }

    def desktop_control(self, action: str, payload_json: str = "{}") -> str:
        self._consume_tool_call()
        try:
            payload = __import__("json").loads(payload_json)
        except Exception:
            payload = {}

        canonical = self._DESKTOP_ACTION_ALIASES.get(action, action)
        # ALLOWED_ACTIONS (src/tools/desktop_control.py) is the real allow-list
        # now -- this used to be an independent if/elif chain that both (a)
        # never checked it, so it wasn't actually enforcing anything, and (b)
        # didn't cover every action ALLOWED_ACTIONS claims (move_mouse had no
        # branch at all, making the advertised capability unreachable).
        if canonical not in _DESKTOP_ALLOWED_ACTIONS or canonical not in _DESKTOP_DISPATCH:
            return f"Unknown or disallowed desktop action: {action}"

        fn = _DESKTOP_DISPATCH[canonical]
        kwargs: dict = {}
        if canonical in {"click", "move_mouse"}:
            kwargs = {"x": int(payload.get("x", 0)), "y": int(payload.get("y", 0))}
        elif canonical in {"type_text", "desktop_send"}:
            kwargs = {"text": str(payload.get("text", ""))}
        elif canonical == "key_press":
            kwargs = {"keys": str(payload.get("key", ""))}

        result = fn(**kwargs)
        if hasattr(result, "message"):
            return result.message
        return str(result)
