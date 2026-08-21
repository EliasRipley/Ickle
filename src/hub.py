import argparse
import json
import keyboard
import threading
from pathlib import Path

from src.feedback_store import record_feedback
from src.icklization import ick
from src.ilm_memory import get_memory
from src.state_store import ILMStateStore
from src.skill_repair import SkillRepairManager


HUB_SEED_EXAMPLES = [
    {"prompt": "Help me make a 1-day Minecraft beginner plan.", "response_style": "step-by-step checklist"},
    {"prompt": "Find social media trends for AI note tools.", "response_style": "short research digest + 3 post ideas"},
    {"prompt": "If unsure, ask me clarifying questions first.", "response_style": "clarification-first behavior"},
]


class LocalHub:
    """Modular command hub for talking to the local assistant + tools."""

    def __init__(self, feedback_path: str = "data/hub_feedback.jsonl"):
        from src.agent import LocalAgent

        self.agent = LocalAgent()
        self.feedback_path = Path(feedback_path)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = ILMStateStore()
        self.memory = get_memory()
        self.repair = SkillRepairManager()
        self.shutdown_requested = False
        self._setup_f12_shutdown()

    def _setup_f12_shutdown(self):
        """Setup F12 key monitoring for emergency shutdown."""
        def on_f12():
            self.shutdown_requested = True
            print("\n\nðŸ›‘ EMERGENCY SHUTDOWN ACTIVATED (F12 pressed)")
            print("Terminating Ickle Hub...")
        
        try:
            keyboard.add_hotkey('f12', on_f12)
            print("âœ“ F12 emergency shutdown enabled")
        except Exception as e:
            print(f"âš  Could not enable F12 shutdown: {e}")
            print("  You may need to run as administrator or use /exit command")

    def run(self):
        print("Ickle Hub ready. Type /help for commands.")
        while True:
            if self.shutdown_requested:
                print("Emergency shutdown completed. Goodbye!")
                break
                
            try:
                user = input("hub> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
                
            if not user:
                continue
            if user in {"/exit", "/quit"}:
                print("bye")
                break
            if user == "/help":
                self._print_help()
                continue
            self._dispatch(user)

    def _print_help(self):
        print(
            "Commands:\n"
            "  /clarify <prompt>\n"
            "  /news <query>\n"
            "  /minecraft-topics\n"
            "  /minecraft <topic_key>\n"
            "  /web <url>\n"
            "  /note <text>\n"
            "  /mode <balanced|direct|power-user>\n"
            "  /policy\n"
            "  /can <task>\n"
            "  /remember <key>=<value>\n"
            "  /recall <key>\n"
            "  /improve <note>\n"
            "  /improvements\n"
            "  /research-find <query>\n"
            "  /research-sessions\n"
            "  /tools\n"
            "  /tool <name> <json-payload>\n"
            "  /cloud-status\n"
            "  /assist <prompt>\n"
            "  /report-failure <skill>|<failure_type>|<details>\n"
            "  /repair-plan <skill>\n"
            "  /scaffold-tool <tool_name>|<description>\n"
            "  /feedback <rating 1-5>|<prompt>|<response>|<notes>\n"
            "  /help, /quit\n"
            "\n"
            "Emergency: Press F12 for immediate shutdown"
        )

    def _dispatch(self, user: str):
        if user.startswith("/clarify "):
            result = self.agent.maybe_request_clarification(user[len("/clarify ") :])
            print(result.question if result.needs_clarification else "Prompt looks specific enough.")
            return

        if user.startswith("/news "):
            print(self.agent.research_marketing_topic(user[len("/news ") :]))
            return

        if user == "/minecraft-topics":
            print(", ".join(self.agent.list_minecraft_topics()))
            return

        if user.startswith("/minecraft "):
            print(self.agent.read_minecraft_topic(user[len('/minecraft ') :].strip()))
            return

        if user.startswith("/web "):
            request = user[len("/web ") :]
            print(self.agent.read_webpage(request))
            return

        if user.startswith("/note "):
            request = user[len("/note ") :]
            path = self.agent.write_note(request)
            print(f"wrote note: {path}")
            return

        if user.startswith("/can "):
            print(self.agent.capability_check(user[len("/can ") :]))
            return

        if user.startswith("/remember "):
            payload = user[len("/remember ") :].strip()
            if "=" not in payload:
                print("Use /remember <key>=<value>")
                return
            key, value = payload.split("=", 1)
            self.state.set_preference(key.strip(), value.strip())
            print(f"Saved preference: {key.strip()}")
            return

        if user.startswith("/recall "):
            key = user[len("/recall ") :].strip()
            value = self.state.get_preference(key)
            print(value if value else "No saved value for that key.")
            return

        if user.startswith("/improve "):
            note = user[len("/improve ") :].strip()
            self.state.add_improvement_note(note)
            print("Saved improvement note.")
            return

        if user == "/improvements":
            notes = self.state.list_improvements()
            if not notes:
                print("No improvement notes saved yet.")
            else:
                for i, note in enumerate(notes, start=1):
                    print(f"{i}. {note}")
            return

        if user.startswith("/research-find "):
            query = user[len("/research-find ") :].strip()
            rows = self.memory.search_research_notes(query, limit=8)
            if not rows:
                print("No matching research notes found.")
                return
            for idx, row in enumerate(rows, start=1):
                src = row.get("source_title") or row.get("source_url") or "source-unknown"
                print(f"{idx}. [{row.get('topic', 'general')}] {row.get('finding', '')} (source: {src})")
            return

        if user == "/research-sessions":
            sessions = self.memory.list_research_sessions(limit=20)
            if not sessions:
                print("No research sessions yet.")
                return
            for idx, session in enumerate(sessions, start=1):
                print(
                    f"{idx}. {session.get('session_id')} topic={session.get('topic')} "
                    f"notes={session.get('note_count', 0)} updated={session.get('updated_at_utc')}"
                )
            return

        if user == "/tools":
            tools = self.agent.list_user_tools()
            print(", ".join(tools) if tools else "No user tools found in ./user_tools")
            return

        if user.startswith("/tool "):
            rest = user[len("/tool ") :].strip()
            if not rest:
                print("Use /tool <name> <json-payload>")
                return
            name, _, payload = rest.partition(" ")
            payload = payload.strip() or "{}"
            print(self.agent.run_user_tool(name=name, payload_json=payload))
            return

        if user == "/cloud-status":
            print(self.agent.cloud_status())
            return

        if user.startswith("/assist "):
            prompt = user[len("/assist ") :].strip()
            print(self.agent.cloud_assist(prompt))
            return

        if user.startswith("/report-failure "):
            payload = user[len("/report-failure ") :].strip()
            parts = payload.split("|", 2)
            if len(parts) != 3:
                print("Use /report-failure <skill>|<failure_type>|<details>")
                return
            skill, failure_type, details = [p.strip() for p in parts]
            self.repair.record_incident(skill, failure_type, details)
            print("Incident recorded.")
            return

        if user.startswith("/repair-plan "):
            skill = user[len("/repair-plan ") :].strip()
            incident = self.repair.latest_for_skill(skill)
            if not incident:
                print("No incidents recorded for that skill.")
                return
            plan = self.repair.plan(incident.failure_type)
            print(f"Latest failure type: {incident.failure_type}")
            for i, action in enumerate(plan, start=1):
                print(f"{i}. {action.step} â€” {action.why}")
            return

        if user.startswith("/scaffold-tool "):
            payload = user[len("/scaffold-tool ") :].strip()
            parts = payload.split("|", 1)
            if len(parts) != 2:
                print("Use /scaffold-tool <tool_name>|<description>")
                return
            name, desc = [p.strip() for p in parts]
            print(self.repair.scaffold_tool(name, desc))
            return

        if user.startswith("/feedback "):
            self._handle_feedback(user[len("/feedback ") :])
            return

        if user.startswith("/mode "):
            self.agent.set_autonomy_mode(user[len("/mode ") :].strip())
            print(self.agent.get_policy_summary())
            return

        if user == "/policy":
            print(self.agent.get_policy_summary())
            return

        print("Unknown command. Use /help.")

    def _handle_feedback(self, payload: str):
        parts = payload.split("|", 3)
        if len(parts) < 4:
            print("Invalid format. Use: /feedback <rating 1-5>|<prompt>|<response>|<notes>")
            return
        rating = int(parts[0].strip())
        record_feedback(
            prompt=parts[1].strip(),
            response=parts[2].strip(),
            rating=rating,
            notes=parts[3].strip(),
            path=str(self.feedback_path),
        )
        print(f"Saved feedback to {self.feedback_path}")


def export_seed(path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in HUB_SEED_EXAMPLES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote hub seed examples: {out}")


def main():
    parser = argparse.ArgumentParser(description="Ickle local modular hub")
    parser.add_argument("--feedback-path", default="data/hub_feedback.jsonl")
    parser.add_argument("--export-seed", default="")
    args = parser.parse_args()

    if args.export_seed:
        export_seed(args.export_seed)
        return

    hub = LocalHub(feedback_path=args.feedback_path)
    hub.run()


if __name__ == "__main__":
    main()
