"""Lazy command routing for optional LIBERO hardware experiments."""

import argparse
import importlib

COMMANDS = {
    "worker": ("worker", ()),
    "campaign": ("campaign", ()),
    "analyze": ("reporting", ("analyze",)),
    "report": ("reporting", ("campaign",)),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        commands.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    module, prefix = COMMANDS[args.command]
    target = importlib.import_module(f"coexecution.profiling.{module}")
    return target.main([*prefix, *remaining])


if __name__ == "__main__":
    main()
