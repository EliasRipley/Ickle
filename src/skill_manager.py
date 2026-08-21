import argparse
import subprocess

from src.icklization import ick
from src.resource_defaults import add_resource_pct_args
from src.skill_system import SkillCard, SkillRegistry


def cmd_list(reg: SkillRegistry):
    skills = reg.list_skills()
    print("\n".join(skills) if skills else "No skills registered.")


def cmd_show(reg: SkillRegistry, name: str):
    card = reg.get_skill(name)
    if not card:
        print(f"Skill not found: {name}")
        return
    print(card)


def cmd_learn(reg: SkillRegistry, name: str, corpus: str, out_model: str, steps: int, cpu_pct: int, ram_pct: int, gpu_pct: int, dry_run: bool):
    activation_prompt = (
        f"You are operating with skill '{name}'. Prioritize this skill's style/constraints while staying truthful about limits."
    )
    card = SkillCard(
        name=name,
        description=f"User-defined learned skill: {name}",
        corpus_path=corpus,
        model_path=out_model,
        activation_prompt=activation_prompt,
    )

    if dry_run:
        print(f"[dry-run] would train skill '{name}' from corpus '{corpus}' -> '{out_model}'")
    else:
        cmd = [
            "python",
            "-m",
            "src.train",
            "--data",
            corpus,
            "--out",
            out_model,
            "--steps",
            str(steps),
            "--cpu-pct",
            str(cpu_pct),
            "--ram-pct",
            str(ram_pct),
            "--gpu-pct",
            str(gpu_pct),
        ]
        subprocess.run(cmd, check=True)

    reg.register_skill(card)
    print(f"Skill registered: {name}")


def main():
    parser = argparse.ArgumentParser(description="ILM skill lifecycle manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    show = sub.add_parser("show")
    show.add_argument("name")

    learn = sub.add_parser("learn")
    learn.add_argument("--name", required=True)
    learn.add_argument("--corpus", required=True)
    learn.add_argument("--out-model", required=True)
    learn.add_argument("--steps", type=int, default=1200)
    add_resource_pct_args(learn)
    learn.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    reg = SkillRegistry()

    if args.cmd == "list":
        cmd_list(reg)
    elif args.cmd == "show":
        cmd_show(reg, args.name)
    elif args.cmd == "learn":
        cmd_learn(reg, args.name, args.corpus, args.out_model, args.steps, args.cpu_pct, args.ram_pct, args.gpu_pct, args.dry_run)


if __name__ == "__main__":
    main()
