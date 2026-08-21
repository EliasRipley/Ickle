from __future__ import annotations

# This is an application module despite its historical filename. Prevent
# pytest from mistaking its helper classes for test cases during discovery.
__test__ = False

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class RepairStep:
    attempt: int
    action: str
    timestamp: str
    result: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairTrajectory:
    trajectory_id: str
    target_file: str
    initial_test_output: str
    steps: list[RepairStep] = field(default_factory=list)
    final_test_passed: bool = False
    resolution: str = ""


@dataclass
class TestFailure:
    test_name: str
    error_type: str
    error_message: str
    file_path: str = ""
    line: int = 0
    traceback: str = ""


def parse_pytest_failures(output: str) -> list[TestFailure]:
    """Parse pytest output to extract structured failure info."""
    failures: list[TestFailure] = []
    lines = output.splitlines()

    error_type = ""
    error_message = ""
    file_path = ""
    line = 0
    in_failure = False
    tb_lines: list[str] = []

    for i, text in enumerate(lines):
        if text.startswith("FAILURES") or text.startswith("==== FAILURES") or text.startswith("FAILED"):
            in_failure = True
            continue
        if in_failure:
            tb_lines.append(text)
            if not error_type:
                match = re.match(r"E\s+(\w+Error|AssertionError)[:\s]*(.*)", text)
                if match:
                    error_type = match.group(1)
                    error_message = match.group(2)
            if not file_path:
                m = re.match(r"E?\s*File\s+['\"]?([^'\"]+)['\"]?[,\s]+line\s+(\d+)", text)
                if m:
                    file_path = m.group(1)
                    line = int(m.group(2))
            if text.strip() == "" and tb_lines:
                failures.append(TestFailure(
                    test_name="",
                    error_type=error_type,
                    error_message=error_message[:500],
                    file_path=file_path,
                    line=line,
                    traceback="\n".join(tb_lines[-15:])[:2000],
                ))
                error_type = ""
                error_message = ""
                file_path = ""
                line = 0
                tb_lines = []
                in_failure = False

    if error_type and not failures:
        failures.append(TestFailure(
            test_name="",
            error_type=error_type,
            error_message=error_message[:500],
            file_path=file_path,
            line=line,
            traceback="\n".join(tb_lines[-15:])[:2000],
        ))

    for i, text in enumerate(lines):
        m = re.match(r"^test_.+?\s+(FAILED|ERROR)", text)
        if m:
            test_name = text.split()[0]
            failure = next((f for f in failures if not f.test_name), None)
            if failure:
                failure.test_name = test_name

    return failures


def parse_jest_failures(output: str) -> list[TestFailure]:
    """Parse Jest/Vitest output to extract structured failure info."""
    failures: list[TestFailure] = []
    lines = output.splitlines()
    current = TestFailure(test_name="", error_type="", error_message="", traceback="")

    for text in lines:
        if "FAIL" in text and "tests/" in text:
            parts = text.split()
            for p in parts:
                if p.endswith(".test.") or p.endswith(".spec."):
                    current = TestFailure(test_name=p, error_type="FAIL", error_message="", traceback="", file_path=p)
                    continue
        if "Expected:" in text or "Received:" in text:
            current.error_type = "AssertionError"
            current.error_message = text.strip()[:500]
            current.traceback += text + "\n"
        elif "error TS" in text:
            current.error_type = "TypeError"
            current.error_message = text.strip()[:500]
        elif "Error:" in text and not current.error_type:
            current.error_type = "Error"
            current.error_message = text.split("Error:", 1)[1].strip()[:500]

    if current.test_name or current.error_message:
        failures.append(current)
    return failures


def run_test_suite(cmd: str = "pytest", cwd: str = ".", timeout_seconds: int = 180) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=max(1, timeout_seconds),
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    return {
        "returncode": result.returncode,
        "output": combined,
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-2000:],
        "passed": result.returncode == 0,
    }


def generate_hypothesis(failure: TestFailure, source_code: str, file_path: str) -> str:
    """Generate a diagnostic hypothesis from a test failure."""
    parts: list[str] = []

    if "ImportError" in failure.error_type or "ModuleNotFound" in failure.error_type:
        parts.append(f"IMPORT ISSUE in {file_path}: {failure.error_message}")
        parts.append("Check that all imported modules are installed and available.")
    elif "AttributeError" in failure.error_type:
        parts.append(f"ATTRIBUTE ACCESS in {file_path}: {failure.error_message}")
        parts.append(f"Target line: {failure.line}")
        ref = re.search(r"has no attribute ['\"](\w+)['\"]", failure.error_message)
        if ref:
            parts.append(f"Missing attribute/function: '{ref.group(1)}'")
            if ref.group(1) in source_code:
                parts.append("Found reference in source â€” may be a typo or order-of-definition issue.")
            else:
                parts.append(f"'{ref.group(1)}' not found in source â€” may need to be imported or defined.")
    elif "TypeError" in failure.error_type:
        parts.append(f"TYPE ERROR in {file_path}: {failure.error_message}")
        parts.append(f"Target line: {failure.line}")
    elif "AssertionError" in failure.error_type or "assert" in failure.error_message.lower():
        parts.append(f"ASSERTION FAILURE in {file_path}: {failure.error_message}")
        parts.append(f"Target line: {failure.line}")
        if failure.line > 0:
            source_lines = source_code.splitlines()
            if failure.line <= len(source_lines):
                parts.append(f"Code at line {failure.line}: {source_lines[failure.line - 1][:120]}")
    elif "SyntaxError" in failure.error_type:
        parts.append(f"SYNTAX ERROR in {file_path}: {failure.error_message}")
    else:
        parts.append(f"ERROR in {file_path}:{failure.line}: {failure.error_type}: {failure.error_message}")

    return "\n".join(parts)


class TestRepairLoop:
    def __init__(self, workspace: str = ".", test_command: str = "pytest"):
        self.workspace = Path(workspace).resolve()
        self.test_command = test_command
        self.trajectories: list[RepairTrajectory] = []

    def diagnose(self, test_output: str, file_path: str) -> list[TestFailure]:
        failures = parse_pytest_failures(test_output)
        if not failures:
            failures = parse_jest_failures(test_output)
        for f in failures:
            if f.file_path and not Path(f.file_path).is_absolute():
                f.file_path = str(self.workspace / f.file_path)
            if not f.file_path:
                f.file_path = file_path
        return failures

    def read_source(self, file_path: str) -> str:
        p = Path(file_path)
        if not p.is_absolute():
            p = self.workspace / file_path
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors="replace")

    def repair_cycle(
        self,
        file_path: str,
        apply_patch: Callable[[str, str, str], bool],
        max_rounds: int = 5,
        timeout_seconds: int = 180,
    ) -> RepairTrajectory:
        tid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]
        trajectory = RepairTrajectory(
            trajectory_id=tid,
            target_file=file_path,
            initial_test_output="",
        )

        for round_idx in range(max_rounds):
            t0 = time.monotonic()
            test_result = run_test_suite(self.test_command, str(self.workspace), timeout_seconds)

            if round_idx == 0:
                trajectory.initial_test_output = test_result["output"][:4000]

            if test_result["passed"]:
                trajectory.final_test_passed = True
                trajectory.resolution = f"All tests passed at round {round_idx + 1}"
                trajectory.steps.append(RepairStep(
                    attempt=round_idx + 1,
                    action="test",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    result="PASSED",
                ))
                break

            failures = self.diagnose(test_result["output"], file_path)
            source = self.read_source(file_path)

            if not failures:
                trajectory.resolution = "Cannot parse failures â€” manual intervention needed"
                trajectory.steps.append(RepairStep(
                    attempt=round_idx + 1,
                    action="diagnose",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    result="Cannot parse failures",
                    metadata={"output_snippet": test_result["output"][:1000]},
                ))
                break

            hypothesis = generate_hypothesis(failures[0], source, file_path)

            trajectory.steps.append(RepairStep(
                attempt=round_idx + 1,
                action="diagnose",
                timestamp=datetime.now(timezone.utc).isoformat(),
                result=f"Found {len(failures)} failure(s)",
                metadata={
                    "failures": [{  # pyright: ignore[reportUnknownVariableType]
                        "test_name": f.test_name,
                        "error_type": f.error_type,
                        "error_message": f.error_message[:200],
                        "line": f.line,
                    } for f in failures],
                    "hypothesis": hypothesis[:1000],
                },
            ))

        if not trajectory.final_test_passed and trajectory.resolution == "":
            trajectory.resolution = f"Failed after {max_rounds} rounds"

        self.trajectories.append(trajectory)
        return trajectory

    def run_and_learn_cycle(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        apply_patch: Callable[[str, str, str], bool],
        max_rounds: int = 5,
    ) -> RepairTrajectory:
        p = Path(file_path)
        if p.exists():
            p.write_text(old_content, encoding="utf-8")

        trajectory = self.repair_cycle(file_path, apply_patch, max_rounds)

        if not trajectory.final_test_passed:
            p.write_text(old_content, encoding="utf-8")
            trajectory.resolution = "Rolled back to original â€” tests still failing"

        return trajectory

    def export_trajectory(self, trajectory: RepairTrajectory, out_path: str):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trajectory_id": trajectory.trajectory_id,
            "target_file": trajectory.target_file,
            "initial_test_output": trajectory.initial_test_output[:5000],
            "final_test_passed": trajectory.final_test_passed,
            "resolution": trajectory.resolution,
            "num_steps": len(trajectory.steps),
            "steps": [
                {
                    "attempt": s.attempt,
                    "action": s.action,
                    "timestamp": s.timestamp,
                    "result": s.result,
                    "metadata": s.metadata,
                }
                for s in trajectory.steps
            ],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def export_trajectories_jsonl(self, out_path: str, max_per_file: int = 50) -> int:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with out.open("w", encoding="utf-8") as f:
            for t in self.trajectories[-max_per_file:]:
                row = {
                    "trajectory_id": t.trajectory_id,
                    "target_file": t.target_file,
                    "passed": t.final_test_passed,
                    "resolution": t.resolution,
                    "num_steps": len(t.steps),
                    "initial_output": t.initial_test_output[:2000],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        return written

    def stats(self) -> dict[str, Any]:
        if not self.trajectories:
            return {"total": 0}
        passed = sum(1 for t in self.trajectories if t.final_test_passed)
        total_steps = sum(len(t.steps) for t in self.trajectories)
        return {
            "total_trajectories": len(self.trajectories),
            "pass_rate": round(passed / max(1, len(self.trajectories)), 4),
            "total_steps": total_steps,
            "avg_steps": round(total_steps / max(1, len(self.trajectories)), 1),
        }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Test repair loop â€” execute tests, diagnose failures, repair, retry")
    ap.add_argument("--workspace", default=".", help="Project workspace root")
    ap.add_argument("--test-cmd", default="pytest", help="Test command to run")
    ap.add_argument("--file", help="Target file to diagnose/repair")
    ap.add_argument("--diagnose", action="store_true", help="Parse test output and show failures")
    ap.add_argument("--test-output", help="Raw test output to parse")
    ap.add_argument("--export", default="data/repair_trajectories.jsonl")
    args = ap.parse_args()

    rl = TestRepairLoop(workspace=args.workspace, test_command=args.test_cmd)

    if args.diagnose and args.test_output:
        output = Path(args.test_output).read_text(encoding="utf-8") if Path(args.test_output).exists() else args.test_output
        failures = rl.diagnose(output, args.file or "")
        for f in failures:
            print(f"[{f.error_type}] {f.test_name}: {f.error_message[:120]}")
            if f.line:
                print(f"  Line {f.line} in {f.file_path}")
        print(f"\nTotal failures: {len(failures)}")
    else:
        test_result = run_test_suite(rl.test_command, str(rl.workspace))
        if test_result["passed"]:
            print("All tests pass â€” no repair needed.")
        else:
            failures = rl.diagnose(test_result["output"], args.file or "")
            print(f"Found {len(failures)} failure(s):")
            for f in failures[:5]:
                print(f"  [{f.error_type}] {f.test_name or 'unknown'}: {f.error_message[:120]}")


if __name__ == "__main__":
    main()
