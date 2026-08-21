import argparse

from src.cloud_assist import assist, cloud_status_text


def main():
    parser = argparse.ArgumentParser(description="Optional cloud assist for ILM")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    if args.status:
        print(cloud_status_text())
        return

    if not args.prompt:
        raise SystemExit("Provide --prompt or use --status")

    model = args.model or None
    print(assist(prompt=args.prompt, model=model))


if __name__ == "__main__":
    main()
