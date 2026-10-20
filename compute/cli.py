"""Stable compute and Slurm command surface."""

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
    parser = argparse.ArgumentParser(prog="thesis-compute")
    parser.add_argument(
        "command", choices=("capacity", "resources", "resume", "continuation", "worker")
    )
    args, rest = parser.parse_known_args()
    command_main: Callable[[], object]
    if args.command == "capacity":
        from compute.slurm.capacity import main as command_main
    elif args.command == "resources":
        from compute.slurm.resources import main as command_main
    elif args.command == "resume":
        from compute.slurm.resume import main as command_main
    elif args.command == "continuation":
        from compute.slurm.continuation import main as command_main
    else:
        from compute.slurm.worker_dispatch import main as command_main
    _forward(command_main, rest)
