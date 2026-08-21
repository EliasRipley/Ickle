import argparse

from src.ilm_profile import detect_resources, ResourceConfig
from src.resource_defaults import add_resource_pct_args


def main():
    parser = argparse.ArgumentParser(description="Show resource configuration for the current system")
    add_resource_pct_args(parser)
    args = parser.parse_args()

    rc = detect_resources()

    if args.cpu_pct and 10 <= args.cpu_pct <= 100:
        rc.cpu_percent = args.cpu_pct
    if args.ram_pct and 10 <= args.ram_pct <= 100:
        rc.ram_percent = args.ram_pct
    if args.gpu_pct:
        rc.gpu_percent = args.gpu_pct

    print(rc.summary())


if __name__ == "__main__":
    main()
