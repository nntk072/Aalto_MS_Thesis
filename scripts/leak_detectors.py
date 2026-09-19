"""CLI for thesis leak detectors (T-03.3).

Example:
    uv run python scripts/leak_detectors.py \\
        --features-csv path/to/feat.csv --target-col ret_1 --feature-col rsi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.evaluation.leakage import (  # noqa: E402
    label_shuffle_score,
    time_shift_feature_scores,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-csv", required=True)
    p.add_argument("--target-col", required=True)
    p.add_argument("--feature-col", required=True)
    p.add_argument("--out", default="results/leak_detectors.json")
    p.add_argument("--n-perm", type=int, default=64)
    return p.parse_args()


def main() -> None:
    """Run label-shuffle and time-shift detectors on two columns."""
    args = parse_args()
    df = pd.read_csv(args.features_csv, index_col=0, parse_dates=True)
    y = df[args.target_col]
    x = df[args.feature_col]
    report = {
        "label_shuffle": label_shuffle_score(y, x, n_perm=args.n_perm),
        "time_shift": time_shift_feature_scores(x, y),
        "target_col": args.target_col,
        "feature_col": args.feature_col,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
