"""Inspect or evaluate the reviewed physical moving-object benchmark."""

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
    evaluate = commands.add_parser(
        "evaluate", help="Run default GR00T on real moving-object environments"
    )
    from .evaluation import add_evaluate_arguments

    add_evaluate_arguments(evaluate)
    streaming = commands.add_parser(
        "evaluate-pir2",
        help="Run existing πR² policies in real moving worlds",
        description=(
            "Evaluate moving-world πR² with native per-request latency by default; "
            "fixed counterfactual timing is an explicit control. "
            "Real GPU/physics work is serial, with independent logical service lanes."
        ),
    )
    from .streaming_evaluation import add_streaming_arguments

    add_streaming_arguments(streaming)
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate-pir2":
            from .streaming_evaluation import evaluate_streaming

            print(json.dumps(evaluate_streaming(args)))
            return
        if args.command == "evaluate":
            from .evaluation import evaluate_default

            result = evaluate_default(args)
            print(json.dumps(result))
            return
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
