"""Stable evaluation command surface."""

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
    parser = argparse.ArgumentParser(prog="thesis-evaluate")
    parser.add_argument("command", choices=("reanalyze",))
    args, rest = parser.parse_known_args()
    from evaluation.reanalysis.runner import main as command_main

    _forward(command_main, rest)
