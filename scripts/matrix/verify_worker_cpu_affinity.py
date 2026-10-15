#!/usr/bin/env python3
"""Fail closed if Slurm did not apply a campaign worker's assigned CPU mask."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-cpus", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    expected = {int(value) for value in args.expected_cpus.split(",") if value}
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not expected or not command:
        parser.error("expected a nonempty CPU set and command after --")
    actual = set(os.sched_getaffinity(0))
    print(
        f"CPU_AFFINITY expected={sorted(expected)} actual={sorted(actual)}",
        flush=True,
    )
    if actual != expected:
        print(
            "ERROR: Slurm CPU affinity differs from the disjoint campaign assignment",
            file=sys.stderr,
        )
        return 78
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
