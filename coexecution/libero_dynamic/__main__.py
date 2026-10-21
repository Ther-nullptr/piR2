"""Inspect the reviewed LIBERO-Dynamic benchmark without loading a simulator."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    from .catalog import SUITES, build_inventory

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser(
        "inventory", help="Validate and enumerate moving-object conditions"
    )
    inventory.add_argument("--upstream", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="Repeat to select suites; default all 40 tasks",
    )
    args = parser.parse_args(argv)
    try:
        result = build_inventory(args.upstream, suites=args.suite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": result["counts"],
                "status": result["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
