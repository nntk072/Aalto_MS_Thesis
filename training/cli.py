"""Stable training command surface."""

from __future__ import annotations

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
    argv = sys.argv[1:]
    commands = {"train", "backtest", "baselines", "lstm-baseline", "vae"}
    command = argv[0] if argv and argv[0] in commands else "train"
    rest = argv[1:] if argv and argv[0] in commands else argv

    if command == "train":
        from training.runner.training import main as command_main
    elif command == "backtest":
        from training.runner.backtest import main as command_main
    elif command == "baselines":
        from training.runner.baselines import main as command_main
    elif command == "lstm-baseline":
        from training.runner.lstm_baseline import main as command_main
    else:
        from training.runner.vae import main as command_main

    _forward(command_main, rest)
