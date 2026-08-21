from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodeEvalCase:
    name: str
    description: str
    repo_root: str = ""
    target_file: str = ""
    buggy_code: str = ""
    expected_fix: str = ""
    instructions: str = ""
    expected_test_cmd: str = ""


@dataclass
class CodeEvalResult:
    case: CodeEvalCase
    applied_patch: str = ""
    test_passed: bool = False
    lint_passed: bool = False
    edit_precision: float = 0.0
    regression_count: int = 0
    rounds_used: int = 0
    error: str = ""


def compute_edit_precision(predicted: str, expected: str) -> float:
    if not expected or not predicted:
        if not expected and not predicted:
            return 1.0
        return 0.0

    pred_lines = predicted.strip().splitlines()
    exp_lines = expected.strip().splitlines()

    pred_set = set(line.strip() for line in pred_lines if line.strip())
    exp_set = set(line.strip() for line in exp_lines if line.strip())

    if not exp_set:
        return 0.0

    exact = sum(1 for pl, el in zip(pred_lines, exp_lines) if pl == el)
    exact_score = exact / max(1, max(len(pred_lines), len(exp_lines)))

    overlap = len(pred_set & exp_set) / len(exp_set)

    return 0.5 * exact_score + 0.5 * overlap


def compute_edit_precision_v2(predicted: str, expected: str) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    pred_tokens = set(re.findall(r"[a-zA-Z_]\w+", predicted))
    exp_tokens = set(re.findall(r"[a-zA-Z_]\w+", expected))

    if not exp_tokens:
        return 0.0

    tp = len(pred_tokens & exp_tokens)
    fp = len(pred_tokens - exp_tokens)
    fn = len(exp_tokens - pred_tokens)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_diff_quality(predicted: str, expected: str) -> dict[str, float]:
    return {
        "line_precision": round(compute_edit_precision(predicted, expected), 4),
        "token_f1": round(compute_edit_precision_v2(predicted, expected), 4),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


BUILTIN_CODE_BENCHMARKS: list[CodeEvalCase] = [
    CodeEvalCase(
        name="fix_missing_import",
        description="Add missing import statement to fix NameError",
        buggy_code='def greet(name):\n    return f"Hello {name}"\n\nprint(greet("World"))',
        expected_fix='def greet(name):\n    return f"Hello {name}"\n\nif __name__ == "__main__":\n    print(greet("World"))',
        instructions="Fix the code so it runs without error. The print should only execute when run as main.",
        expected_test_cmd="python -c",
    ),
    CodeEvalCase(
        name="fix_off_by_one",
        description="Fix off-by-one error in loop",
        buggy_code="def sum_range(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total",
        expected_fix="def sum_range(n):\n    total = 0\n    for i in range(n + 1):\n        total += i\n    return total",
        instructions="Fix the loop so it includes n in the sum.",
    ),
    CodeEvalCase(
        name="fix_type_error",
        description="Fix type error when adding string to int",
        buggy_code="def add_score(scores, name, score):\n    scores[name] = scores.get(name, 0) + score\n    return scores",
        expected_fix="def add_score(scores, name, score):\n    scores[name] = int(scores.get(name, 0)) + int(score)\n    return scores",
        instructions="Fix the function to safely add scores when they might be strings.",
    ),
    CodeEvalCase(
        name="fix_key_error",
        description="Fix KeyError by using dict.get with default",
        buggy_code="def get_setting(config, key):\n    return config[key]",
        expected_fix="def get_setting(config, key, default=None):\n    return config.get(key, default)",
        instructions="Fix the function to not crash when a key is missing.",
    ),
    CodeEvalCase(
        name="fix_index_error",
        description="Fix IndexError by checking list bounds",
        buggy_code="def safe_get(items, idx):\n    return items[idx]",
        expected_fix="def safe_get(items, idx):\n    if 0 <= idx < len(items):\n        return items[idx]\n    return None",
        instructions="Fix the function to safely access list items without IndexError.",
    ),
]


class CodeEvalRunner:
    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace)
        self.results: list[CodeEvalResult] = []

    def evaluate_fix(
        self,
        case: CodeEvalCase,
        apply_fix: Any,
        run_tests: Any,
        run_lint: Any | None = None,
    ) -> CodeEvalResult:
        result = CodeEvalResult(case=case)

        try:
            applied = apply_fix(case.buggy_code, case.instructions)
            result.applied_patch = str(applied or "")[:4000]
        except Exception as e:
            result.error = f"Fix application failed: {e}"
            self.results.append(result)
            return result

        result.edit_precision = compute_edit_precision(result.applied_patch, case.expected_fix)

        try:
            test_out = run_tests(result.applied_patch)
            result.test_passed = bool(test_out)
        except Exception as e:
            result.error = f"Test execution failed: {e}"

        if run_lint:
            try:
                lint_out = run_lint(result.applied_patch)
                result.lint_passed = bool(lint_out)
            except Exception:
                pass

        self.results.append(result)
        return result

    def run_benchmark(self, apply_fix: Any, run_tests: Any, run_lint: Any | None = None) -> dict[str, Any]:
        self.results.clear()
        for case in BUILTIN_CODE_BENCHMARKS:
            self.evaluate_fix(case, apply_fix, run_tests, run_lint)

        passed = [r for r in self.results if r.test_passed]
        precisions = [r.edit_precision for r in self.results]
        avg_precision = sum(precisions) / max(1, len(precisions))

        return {
            "total_cases": len(BUILTIN_CODE_BENCHMARKS),
            "passed": len(passed),
            "failed": len(self.results) - len(passed),
            "pass_rate": round(len(passed) / max(1, len(self.results)), 4),
            "avg_edit_precision": round(avg_precision, 4),
            "per_case": [
                {
                    "name": r.case.name,
                    "passed": r.test_passed,
                    "linted": r.lint_passed,
                    "precision": round(r.edit_precision, 4),
                    "error": r.error[:200] if r.error else "",
                }
                for r in self.results
            ],
        }

    def run_repair_benchmark(
        self,
        apply_fix_fn: Any,
        run_tests_fn: Any,
        repair_fn: Any,
        max_repair_rounds: int = 3,
    ) -> dict[str, Any]:
        self.results.clear()
        for case in BUILTIN_CODE_BENCHMARKS:
            result = self.evaluate_fix(case, apply_fix_fn, run_tests_fn)
            if result.test_passed:
                result.rounds_used = 1
                continue

            for round_idx in range(1, max_repair_rounds):
                try:
                    repair_out = repair_fn(case.buggy_code, result.applied_patch, case.instructions)
                    result.applied_patch = str(repair_out or "")[:4000]
                    result.edit_precision = compute_edit_precision(result.applied_patch, case.expected_fix)
                    test_out = run_tests_fn(result.applied_patch)
                    result.test_passed = bool(test_out)
                    if result.test_passed:
                        result.rounds_used = round_idx + 1
                        break
                except Exception:
                    continue

        passed = [r for r in self.results if r.test_passed]
        precisions = [r.edit_precision for r in self.results]
        avg_precision = sum(precisions) / max(1, len(precisions))
        avg_rounds = sum(r.rounds_used for r in self.results) / max(1, len(self.results))

        return {
            "total_cases": len(BUILTIN_CODE_BENCHMARKS),
            "first_try_passed": len([r for r in self.results if r.rounds_used == 1 and r.test_passed]),
            "repair_passed": len([r for r in self.results if r.rounds_used > 1 and r.test_passed]),
            "total_passed": len(passed),
            "pass_rate": round(len(passed) / max(1, len(self.results)), 4),
            "avg_rounds": round(avg_rounds, 2),
            "avg_edit_precision": round(avg_precision, 4),
        }


class RegressionTracker:
    def __init__(self, storage_path: str = "data/regression_tracker.json"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            self._data = {"snapshots": {}, "regressions": []}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {"snapshots": {}, "regressions": []}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def record_snapshot(self, label: str, test_results: dict[str, Any]):
        self._data.setdefault("snapshots", {})[label] = {
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "results": test_results,
        }
        self._save()

    def detect_regression(self, current_label: str, baseline_label: str) -> dict[str, Any]:
        snaps = self._data.get("snapshots", {})
        current = snaps.get(current_label, {}).get("results", {})
        baseline = snaps.get(baseline_label, {}).get("results", {})

        curr_passed = current.get("passed", 0)
        base_passed = baseline.get("passed", 0)
        curr_total = current.get("total", 1)
        base_total = baseline.get("total", 1)

        new_failures = curr_passed < base_passed
        regressed = max(0, base_passed - curr_passed)

        regression = {
            "baseline": baseline_label,
            "current": current_label,
            "baseline_pass_rate": round(base_passed / max(1, base_total), 4),
            "current_pass_rate": round(curr_passed / max(1, curr_total), 4),
            "regressed_tests": regressed,
            "is_regression": new_failures,
        }

        if new_failures:
            regressions = self._data.setdefault("regressions", [])
            regressions.append(regression)
            regressions[:] = regressions[-500:]
            self._save()

        return regression

    def get_regression_history(self) -> list[dict[str, Any]]:
        return list(self._data.get("regressions", []))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Code-specific evaluation benchmarks")
    ap.add_argument("--run-benchmark", action="store_true", help="Run the built-in code benchmark")
    ap.add_argument("--workspace", default=".", help="Workspace root")
    ap.add_argument("--export", default="data/code_eval_results.json")
    args = ap.parse_args()

    if args.run_benchmark:
        def dummy_apply(buggy_code, instructions):
            return f"# Fixed: {instructions}\n{buggy_code}"

        def dummy_test(code):
            return True

        evaluator = CodeEvalRunner(workspace=args.workspace)
        report = evaluator.run_benchmark(dummy_apply, dummy_test)
        print(json.dumps(report, indent=2))

        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
