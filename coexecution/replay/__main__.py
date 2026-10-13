"""Build local replay evidence; no model execution or web dependencies."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Normalize explicit recorded model/deployment evidence"
    )
    build.add_argument("--model-run", action="append", type=Path, default=[])
    build.add_argument("--deployment", action="append", type=Path, default=[])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--clock-cohort",
        help="Operator-declared audited common device/clock cohort; omitted disables paired comparison claims",
    )
    args = parser.parse_args()
    from .catalog import build_catalog

    catalog = build_catalog(
        model_runs=args.model_run,
        deployments=args.deployment,
        output=args.output,
        clock_cohort=args.clock_cohort,
    )
    print(f"Built {len(catalog['runs'])} replay runs at {args.output / 'catalog.json'}")


if __name__ == "__main__":
    main()
