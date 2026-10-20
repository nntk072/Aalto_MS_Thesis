"""Stable market-data command surface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable


def _forward(callable_main: Callable[[], object], argv: list[str]) -> None:
    original = sys.argv
    try:
        sys.argv = [original[0], *argv]
        callable_main()
    finally:
        sys.argv = original


def main() -> None:
    parser = argparse.ArgumentParser(prog="thesis-data")
    parser.add_argument("command", choices=("prepare", "discover", "prebuild-campaign-cache"))
    args, rest = parser.parse_known_args()
    if args.command == "prepare":
        from market_data.commands.prepare import main as command_main
    elif args.command == "discover":
        from market_data.commands.discover import main as command_main
    else:
        from market_data.commands.prebuild_campaign_cache import main as command_main
    _forward(command_main, rest)
