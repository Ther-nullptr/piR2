"""Build recorded evidence or explicitly start a live model recording."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Normalize explicit recorded model/deployment evidence"
    )
    build.add_argument("--model-run", action="append", type=Path, default=[])
    build.add_argument("--deployment", action="append", type=Path, default=[])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--default-execution-clock",
        choices=("paced", "ready"),
        help="Explicit robot playback default; must exist in recorded evidence",
    )
    build.add_argument(
        "--default-execution-coupling",
        choices=("asynchronous", "synchronous"),
        help="Explicit robot coupling default; must exist with the selected clock",
    )
    build.add_argument(
        "--clock-cohort",
        help="Operator-declared audited common device/clock cohort; omitted disables paired comparison claims",
    )
    from coexecution.libero.measurement import build_parser

    model_parser = build_parser(add_help=False)
    record = commands.add_parser(
        "record-model",
        parents=[model_parser],
        help="Explicitly run model inference and record a live local stream",
    )
    record.add_argument("--catalog-output", type=Path, required=True)
    record.add_argument("--clock-cohort")
    args = parser.parse_args(argv)
    if args.command == "record-model":
        from .live import record_model

        model_args = argparse.Namespace(
            **{
                action.dest: getattr(args, action.dest)
                for action in model_parser._actions
            }
        )
        try:
            catalog = record_model(
                model_args,
                catalog_output=args.catalog_output,
                clock_cohort=args.clock_cohort,
            )
        except ValueError as error:
            parser.error(str(error))
        print(
            f"Recorded {len(catalog['runs'])} catalog runs at {args.catalog_output / 'catalog.json'}"
        )
        return
    from .catalog import build_catalog

    catalog = build_catalog(
        model_runs=args.model_run,
        deployments=args.deployment,
        output=args.output,
        clock_cohort=args.clock_cohort,
        default_execution_clock=args.default_execution_clock,
        default_execution_coupling=args.default_execution_coupling,
    )
    print(f"Built {len(catalog['runs'])} replay runs at {args.output / 'catalog.json'}")


if __name__ == "__main__":
    main()
