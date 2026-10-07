"""Run the weekly thesis experiment matrix without overwriting evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_rl.orchestration.weekly import STRATEGIES, BatchSpec, WeeklyOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse weekly orchestration options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--objective", default="Weekly strategy comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--config")
    parser.add_argument("--dataset-hash", default="unspecified")
    parser.add_argument("--evaluation-protocol", default="validation_only")
    parser.add_argument("--train-end")
    parser.add_argument("--test-start")
    parser.add_argument("--output-root", default="outputs/thesis_progress")
    parser.add_argument("--retries", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Execute configured weekly batches."""
    args = parse_args()
    if args.batches < 1 or args.retries < 0:
        raise SystemExit("--batches must be positive and --retries must not be negative")
    spec = BatchSpec(
        objective=args.objective,
        strategies=tuple(args.strategies),
        seeds=tuple(args.seeds),
        steps=args.steps,
        config=args.config,
        dataset_hash=args.dataset_hash,
        evaluation_protocol=args.evaluation_protocol,
        train_end=args.train_end,
        test_start=args.test_start,
    )
    week_dir = WeeklyOrchestrator(Path(args.output_root)).run_week(
        args.week, [spec] * args.batches, retries=args.retries
    )
    print(f"Weekly artifacts written to {week_dir}")


if __name__ == "__main__":
    main()
