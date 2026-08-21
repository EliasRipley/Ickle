import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.sqlite_store import SQLiteEventStore


@dataclass
class SkillIncident:
    skill: str
    failure_type: str
    details: str
    timestamp_utc: str


@dataclass
class RepairAction:
    step: str
    why: str


FAILURE_PLAYBOOK = {
    "missing_tool": [
        RepairAction("Create a user tool plugin scaffold", "No executable tool exists for this skill."),
        RepairAction("Register capability alias", "So ILM knows the task is now supported."),
        RepairAction("Add regression test", "Prevent recurring 'supported but broken' claims."),
    ],
    "auth_error": [
        RepairAction("Validate credentials/env vars", "Most social integrations fail due to missing tokens."),
        RepairAction("Add preflight check for auth", "Fail fast with honest message before attempting action."),
    ],
    "rate_limit": [
        RepairAction("Backoff and retry with jitter", "API/platform rate limits are transient."),
        RepairAction("Queue posts and throttle", "Avoid repeated limits and account flags."),
    ],
    "selector_error": [
        RepairAction("Update DOM selectors in tool", "Web UI changed and automation broke."),
        RepairAction("Add page snapshot test", "Catch future selector drift."),
    ],
    "unknown": [
        RepairAction("Capture logs and reproduce", "Need objective signal before retraining/fixing."),
        RepairAction("Add explicit capability limit until fixed", "Avoid hallucinated success claims."),
    ],
}


class SkillRepairManager:
    def __init__(self, root: str = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.incidents_path = self.root / "skill_incidents.jsonl"
        self.event_store = SQLiteEventStore()

    def record_incident(self, skill: str, failure_type: str, details: str):
        incident = SkillIncident(
            skill=skill,
            failure_type=failure_type,
            details=details,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        with self.incidents_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(incident), ensure_ascii=False) + "\n")
        self.event_store.add_event("skill_incident", skill, asdict(incident))

    def plan(self, failure_type: str) -> list[RepairAction]:
        return FAILURE_PLAYBOOK.get(failure_type, FAILURE_PLAYBOOK["unknown"])

    def latest_for_skill(self, skill: str) -> SkillIncident | None:
        if not self.incidents_path.exists():
            return None
        rows = []
        with self.incidents_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("skill") == skill:
                    rows.append(row)
        if not rows:
            return None
        return SkillIncident(**rows[-1])

    def scaffold_tool(self, tool_name: str, description: str) -> str:
        tools_dir = Path("user_tools")
        tools_dir.mkdir(parents=True, exist_ok=True)
        target = tools_dir / f"{tool_name}.py"
        if target.exists():
            return f"Tool already exists: {target}"
        target.write_text(
            "def run(payload):\n"
            f"    # TODO: {description}\n"
            "    return 'NOT IMPLEMENTED: add platform logic here'\n",
            encoding="utf-8",
        )
        return f"Scaffold created: {target}"
