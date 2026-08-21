"""CLI for trainer operator mode â€” submit and run training programs."""

from __future__ import annotations

import argparse
import json

from src.trainer_orchestrator import TrainerOperator, TrainerProgram, operator_from_dict


def main():
    parser = argparse.ArgumentParser(description="Trainer operator â€” run training programs")
    sub = parser.add_subparsers(dest="action", required=True)

    p_submit = sub.add_parser("submit", help="Submit a training program from JSON")
    p_submit.add_argument("--program", required=True, help="Path to JSON program file")
    p_submit.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="Execute a queued program")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--json", action="store_true")

    p_list = sub.add_parser("list", help="List all runs")
    p_list.add_argument("--status", default="", choices=["", "queued", "running", "completed", "failed"])
    p_list.add_argument("--json", action="store_true")

    p_get = sub.add_parser("get", help="Get run details")
    p_get.add_argument("--run-id", required=True)
    p_get.add_argument("--json", action="store_true")

    args = parser.parse_args()
    op = TrainerOperator()

    if args.action == "submit":
        with open(args.program, "r", encoding="utf-8") as f:
            data = json.load(f)
        program = operator_from_dict(data)
        run = op.submit_program(program)
        result = {"run_id": run.run_id, "status": run.status}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Program queued: {run.run_id}")

    elif args.action == "run":
        run = op.execute_run(args.run_id, progress_cb=print)
        result = {
            "run_id": run.run_id, "status": run.status,
            "result": run.result, "error": run.error,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Run {run.run_id}: {run.status}")
            if run.error:
                print(f"Error: {run.error}")

    elif args.action == "list":
        runs = op.list_runs(args.status)
        if args.json:
            print(json.dumps({"runs": runs}, indent=2))
        else:
            for r in runs:
                print(f"  {r['run_id']} [{r['status']}] {r['topic'] or r['output_model']}")

    elif args.action == "get":
        run = op.get_run(args.run_id)
        if not run:
            print(f"Run not found: {args.run_id}")
            return
        data = {
            "run_id": run.run_id, "status": run.status,
            "program": {
                "topic": run.program.topic,
                "model_path": run.program.model_path,
                "output_model": run.program.output_model,
                "steps": run.program.steps,
                "promote_to": run.program.promote_to,
            },
            "steps": run.steps, "result": run.result,
            "created_at": run.created_at, "updated_at": run.updated_at,
            "error": run.error,
        }
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
