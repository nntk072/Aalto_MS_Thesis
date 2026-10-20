"""Stable experiment and campaign command surface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def _forward(callable_main: Callable[[], object], argv: list[str]) -> None:
    original = sys.argv
    try:
        sys.argv = [original[0], *argv]
        result = callable_main()
        if isinstance(result, int) and result:
            raise SystemExit(result)
    finally:
        sys.argv = original


def main() -> None:
    parser = argparse.ArgumentParser(prog="thesis-campaign")
    parser.add_argument(
        "command", choices=("run", "status", "summarize", "checkpoint-map", "variants")
    )
    args, rest = parser.parse_known_args()
    command_main: Callable[[], object]
    if args.command == "run":
        from experiments.campaigns.execution import main as command_main
    elif args.command == "status":
        from experiments.campaigns.status import main as command_main
    elif args.command == "summarize":
        from experiments.campaigns.reporting import main as command_main
    elif args.command == "checkpoint-map":
        from experiments.campaigns.checkpoints import main as command_main
    else:
        from experiments.variants.runner import main as command_main
    _forward(command_main, rest)
