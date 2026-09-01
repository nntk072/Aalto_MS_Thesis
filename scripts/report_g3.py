"""Aggregate RL run metrics and check the Gate G3 criteria.

Reads run reports under --runs-dir in either of two schemas:
  - ``metrics.json`` from ``scripts/train_rl.py`` — ``in_sample`` /
    ``out_of_sample`` blocks (also legacy flat top-level reports); or
  - ``training_log.json`` from ``quant_rl/train/train_rl.py`` — flat
    ``test_sharpe`` / ``test_max_dd`` / ``test_trades`` / ``test_return`` /
    ``test_breaches`` keys (the test-split metrics *are* the OOS block).

Prints a comparison table and evaluates, on the **out-of-sample** metrics
block only:
  - conditional Sharpe > 1.0 on the held-out test slice
  - zero kill-switch breaches

Example:
    python scripts/report_g3.py --runs-dir outputs --sharpe-threshold 1.0
    python scripts/report_g3.py --runs-dir models/rl_runs --sharpe-threshold 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        default="models/rl_runs",
        help="Base dir containing either */metrics.json (scripts/train_rl.py) "
        "or */training_log.json (quant_rl/train/train_rl.py) run reports",
    )
    parser.add_argument("--sharpe-threshold", type=float, default=1.0)
    return parser.parse_args()


def iter_reports(runs_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    """Yield (run_dir, report) for every run under *runs_dir*.

    Prefers ``metrics.json`` when a run dir contains both (the richer,
    in/out-of-sample schema); falls back to flat ``training_log.json``.
    """
    found: list[tuple[Path, dict[str, object]]] = []
    if not runs_dir.is_dir():
        return found
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = run_dir / "metrics.json"
        legacy = run_dir / "training_log.json"
        if metrics.is_file():
            found.append((run_dir, json.loads(metrics.read_text())))
        elif legacy.is_file():
            found.append((run_dir, json.loads(legacy.read_text())))
    return found


def _is_flat_log(run_dir: Path) -> bool:
    """True when the run dir holds only the flat ``training_log.json`` shape."""
    return (run_dir / "training_log.json").is_file() and not (run_dir / "metrics.json").is_file()


def _oos(report: dict[str, object], *, is_flat_log: bool) -> dict[str, object]:
    """Extract the out-of-sample metrics block from either schema.

    For the flat ``training_log.json`` shape the test-split metric keys
    (``test_sharpe`` etc.) *are* the OOS block; legacy metrics.json reports
    with only a top-level ``sharpe`` key fall back to the whole report.
    """
    if not is_flat_log:
        oos = report.get("out_of_sample")
        return oos if isinstance(oos, dict) else report
    return {
        "sharpe": report.get("test_sharpe", 0.0),
        "max_drawdown": report.get("test_max_dd", 0.0),
        "breach_count": report.get("test_breaches", 0),
        "n_trades": report.get("test_trades", 0),
    }


def main() -> None:
    """Load all run reports, print a table and evaluate Gate G3."""
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    reports = iter_reports(runs_dir)
    if not reports:
        raise SystemExit(f"no metrics.json or training_log.json found under {runs_dir}")

    # Normalize every run into a common row: name/algo/arch/vae + oos block.
    rows: list[dict[str, object]] = []
    for run_dir, report in reports:
        is_flat = _is_flat_log(run_dir)
        name = str(report.get("run_name", run_dir.name))
        rows.append(
            {
                "name": name,
                "algo": str(report.get("algo", name.split("_")[0] if "_" in name else "?")),
                "arch": str(report.get("arch", name.split("_")[1] if "_" in name else "?")),
                "vae": "_vae1" in name,
                "oos": _oos(report, is_flat_log=is_flat),
            }
        )

    header = (
        f"{'run':32} {'algo':5} {'arch':12} {'vae':4} {'oos_shr':>8} {'oos_mdd':>8} {'brch':>5}"
    )
    print(header)
    print("-" * len(header))
    conditional_passes: list[bool] = []
    for r in rows:
        name, oos = str(r["name"]), r["oos"]
        sharpe = float(oos.get("sharpe", 0.0))  # type: ignore[arg-type]
        mdd = float(oos.get("max_drawdown", 0.0))  # type: ignore[arg-type]
        breach = int(oos.get("breach_count", 0))  # type: ignore[arg-type]
        print(
            f"{name:32} {r['algo']:5} {r['arch']:12} "
            f"{'yes' if r['vae'] else 'no':4} {sharpe:8.3f} {mdd:8.4f} {breach:5d}"
        )
        if r["vae"]:
            conditional_passes.append(sharpe > args.sharpe_threshold and breach == 0)

    best_sharpe = max(float(r["oos"].get("sharpe", 0.0)) for r in rows)  # type: ignore[arg-type]
    any_breach = any(int(r["oos"].get("breach_count", 0)) > 0 for r in rows)  # type: ignore[arg-type]
    g3 = bool(conditional_passes) and not any_breach and best_sharpe > args.sharpe_threshold
    print("-" * len(header))
    verdict = "PASS" if g3 else "NOT PASSED"
    print(
        f"Gate G3 ({verdict}, out-of-sample): best Sharpe {best_sharpe:.3f} "
        f"(threshold {args.sharpe_threshold}), breaches={'yes' if any_breach else 'no'}"
    )


if __name__ == "__main__":
    main()
