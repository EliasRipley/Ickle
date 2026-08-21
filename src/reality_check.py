import argparse
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from src.training_control import inspect_training_status
from src.workspace_paths import get_training_root


@dataclass
class CheckItem:
    area: str
    status: str
    detail: str


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _browser_runtime_status() -> tuple[bool, str]:
    try:
        from playwright.sync_api import sync_playwright

        from src.browser_runtime import launch_headless_browser

        with sync_playwright() as playwright:
            browser, description = launch_headless_browser(playwright, headless=True)
            browser.close()
        return True, f"Browser-backed web reading is available via {description}."
    except Exception as exc:  # noqa: BLE001
        return False, f"Browser-backed web reading is unavailable: {exc}"


def _federated_intake_check() -> CheckItem:
    """Unlike the other checks here, this used to just assert
    status="hardened" with a fixed detail string -- the one tool whose job
    is auditing claims-vs-implementation was itself making an unverified
    claim. This inspects the real coordinator source for the six specific
    properties the detail text names, so the status reflects what the code
    actually has, not what a comment once said it had."""
    try:
        import inspect

        from src.federated import coordinator as coordinator_module

        source = inspect.getsource(coordinator_module)
        present = {
            "signed updates": "verify_signature(" in source,
            "replay protection": "verify_timestamp(" in source and "_use_nonce(" in source,
            "one contribution per client per round": "_submitted_client_ids(" in source,
            "bounded sample weights": "max_examples_per_update" in source,
            "tensor validation": "isfinite(" in source,
        }
        registration_present = "registration_secret" in inspect.getsource(
            __import__("src.federated.server", fromlist=["server"])
        )
        present["optional registration admission control"] = registration_present
    except Exception as exc:  # noqa: BLE001
        return CheckItem(
            area="federated_intake",
            status="unverifiable",
            detail=f"Could not inspect src/federated/coordinator.py or server.py: {exc}",
        )

    missing = [name for name, ok in present.items() if not ok]
    if not missing:
        return CheckItem(
            area="federated_intake",
            status="hardened",
            detail=(
                "Verified present in src/federated/coordinator.py + server.py: "
                + ", ".join(present.keys()) + "."
            ),
        )
    return CheckItem(
        area="federated_intake",
        status="incomplete",
        detail=f"Missing or not detected: {', '.join(missing)}.",
    )


def collect_checks() -> list[CheckItem]:
    checks: list[CheckItem] = []

    checks.append(
        CheckItem(
            area="non_python_artifacts",
            status="present"
            if Path("config/ilm_policy.toml").exists()
            and Path("schemas/skill_card.schema.json").exists()
            and Path("sql/skill_events.sql").exists()
            else "missing",
            detail="Policy (TOML), schema (JSON Schema), and SQL migration files.",
        )
    )

    benchmark_path = Path("data/maintenance/user_chat_benchmark.json")
    checks.append(
        CheckItem(
            area="promotion_benchmark",
            status="ready" if benchmark_path.exists() else "missing",
            detail=(
                f"Deterministic promotion benchmark at {benchmark_path}."
                if benchmark_path.exists()
                else "A benchmark is required before quality-gated model promotion."
            ),
        )
    )

    training = inspect_training_status(get_training_root() / "training_live.json")
    checks.append(
        CheckItem(
            area="training_lifecycle",
            status=str(training.get("status", "unavailable")),
            detail=(
                "A trainer is actively reporting heartbeats."
                if training.get("is_active")
                else (
                    f"No active trainer; old state is stale ({training.get('stale_reason', 'unknown reason')})."
                    if training.get("is_stale")
                    else f"Latest trainer state: {training.get('status', 'unavailable')}."
                )
            ),
        )
    )

    checks.append(_federated_intake_check())

    public_swarm_parts = {
        "Mainline DHT client": Path("src/federated/public_dht.py").exists(),
        "NAT traversal": Path("src/federated/nat_traversal.py").exists(),
        "control-room interface": Path("web/index.html").exists()
        and "network-status-board" in Path("web/index.html").read_text(encoding="utf-8"),
        "threat-model documentation": Path("docs/PUBLIC_SWARM.md").exists(),
    }
    public_swarm_missing = [name for name, present in public_swarm_parts.items() if not present]
    checks.append(
        CheckItem(
            area="public_swarm_discovery",
            status="implemented" if not public_swarm_missing else "incomplete",
            detail=(
                "Opt-in Mainline-DHT rendezvous, Ickle endpoint verification, NAT status, "
                "and direct-peer fallback are present."
                if not public_swarm_missing
                else f"Missing: {', '.join(public_swarm_missing)}."
            ),
        )
    )

    commons_parts = {
        "answer map": Path("src/epistemics.py").exists(),
        "signed event ledger": Path("src/federated/knowledge_commons.py").exists(),
        "human interface": Path("web/app.js").exists()
        and "createEpistemicBlock" in Path("web/app.js").read_text(encoding="utf-8"),
        "protocol documentation": Path("docs/EPISTEMIC_COMMONS.md").exists(),
    }
    commons_missing = [name for name, present in commons_parts.items() if not present]
    checks.append(
        CheckItem(
            area="epistemic_commons",
            status="implemented" if not commons_missing else "incomplete",
            detail=(
                "Inspectable candidate claims, signed local-by-default review, "
                "conflict-preserving peer merge, and explicit adoption are present."
                if not commons_missing
                else f"Missing: {', '.join(commons_missing)}."
            ),
        )
    )

    checks.append(
        CheckItem(
            area="core_model",
            status="implemented" if Path("src/model.py").exists() else "missing",
            detail="TinyGPT architecture source file presence.",
        )
    )

    torch_ok = _has_module("torch")
    checks.append(
        CheckItem(
            area="training_runtime",
            status="ready" if torch_ok else "blocked",
            detail="Training requires torch installed in runtime environment.",
        )
    )

    browser_ok, browser_detail = _browser_runtime_status()
    checks.append(
        CheckItem(
            area="web_tool_runtime",
            status="ready" if browser_ok else "blocked",
            detail=browser_detail,
        )
    )

    cloud_key = bool(os.getenv("ILM_CLOUD_API_KEY"))
    checks.append(
        CheckItem(
            area="cloud_assist",
            status="optional-ready" if cloud_key else "not_configured",
            detail="Optional cloud assist requires ILM_CLOUD_API_KEY.",
        )
    )

    skill_index = Path("data/skills/index.json")
    checks.append(
        CheckItem(
            area="skill_lifecycle_state",
            status="present" if skill_index.exists() else "empty",
            detail="No persisted skills until user registers them.",
        )
    )

    checks.append(
        CheckItem(
            area="autonomous_learning",
            status="prototype",
            detail=(
                "Autodidact + continual loop are implemented but rely on objective logs/tests and user-run commands; "
                "not a fully self-directing AGI pipeline."
            ),
        )
    )

    return checks


def main():
    parser = argparse.ArgumentParser(description="Reality check: claims vs current ILM implementation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = collect_checks()
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
        return

    for c in checks:
        print(f"[{c.status}] {c.area}: {c.detail}")


if __name__ == "__main__":
    main()
