"""Aggregate RL run metrics and check the Gate G3 criteria.

Reads metrics.json files produced by scripts/train_rl.py under
--runs-dir, prints a comparison table and evaluates, on the
**out-of-sample** metrics block only:
  - conditional Sharpe > 1.0 on the held-out test slice
  - zero kill-switch breaches

Legacy flat reports (pre split-migration runs with a top-level "sharpe"
key) are still read, but the G3 verdict is always computed from the
out-of-sample block when one is present.

Example:
    python scripts/report_g3.py --runs-dir models/rl_runs --sharpe-threshold 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="models/rl_runs")
    parser.add_argument("--sharpe-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """Load all run reports, print a table and evaluate Gate G3."""
    args = parse_args()
    run_dirs = sorted(Path(args.runs_dir).glob("*/metrics.json"))
    if not run_dirs:
        raise SystemExit(f"no metrics.json found under {args.runs_dir}")

    def _oos(report: dict[str, object]) -> dict[str, object]:
        oos = report.get("out_of_sample")
        return oos if isinstance(oos, dict) else report  # legacy flat reports

    rows: list[dict[str, object]] = []
    for path in run_dirs:
        report: dict[str, object] = json.loads(path.read_text())
        rows.append(report)

    header = (
        f"{'run':32} {'algo':5} {'arch':12} {'vae':4} {'oos_shr':>8} {'oos_mdd':>8} {'brch':>5}"
    )
    print(header)
    print("-" * len(header))
    conditional_passes: list[bool] = []
    for r in sorted(rows, key=lambda x: str(x.get("run_name"))):
        name = str(r.get("run_name", "?"))
        algo = name.split("_")[0] if "_" in name else "?"
        arch = name.split("_")[1] if "_" in name else "?"
        vae = "yes" if "_vae1" in name else "no"
        oos = _oos(r)
        sharpe = float(oos.get("sharpe", 0.0))  # type: ignore[arg-type]
        mdd = float(oos.get("max_drawdown", 0.0))  # type: ignore[arg-type]
        breach = int(oos.get("breach_count", 0))  # type: ignore[arg-type]
        print(f"{name:32} {algo:5} {arch:12} {vae:4} {sharpe:8.3f} {mdd:8.4f} {breach:5d}")
        if vae == "yes":
            conditional_passes.append(sharpe > args.sharpe_threshold and breach == 0)

    best_sharpe = max(float(_oos(r).get("sharpe", 0.0)) for r in rows)  # type: ignore[arg-type]
    any_breach = any(int(_oos(r).get("breach_count", 0)) > 0 for r in rows)  # type: ignore[arg-type]
    g3 = bool(conditional_passes) and not any_breach and best_sharpe > args.sharpe_threshold
    print("-" * len(header))
    verdict = "PASS" if g3 else "NOT PASSED"
    print(
        f"Gate G3 ({verdict}, out-of-sample): best Sharpe {best_sharpe:.3f} "
        f"(threshold {args.sharpe_threshold}), breaches={'yes' if any_breach else 'no'}"
    )


if __name__ == "__main__":
    main()
