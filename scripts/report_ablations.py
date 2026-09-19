"""Aggregate ablation JSON reports into a comparison table.

Reads ``results/ablations/*.json`` produced by ``scripts/ablation_runner.py``
and prints a markdown-friendly table of mean OOS metrics (Sharpe, MaxDD,
trades, win rate, key ``pnl_dist_*`` extras when present).

Example:
    python scripts/report_ablations.py --ablations-dir results/ablations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablations-dir",
        default="results/ablations",
        help="Directory of per-variant *.json reports from ablation_runner",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the comparison table as JSON",
    )
    return parser.parse_args()


def load_ablation_reports(ablations_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` ablation report under *ablations_dir*."""
    if not ablations_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(ablations_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            continue
        extras_raw = raw.get("extras")
        extras: dict[str, Any] = extras_raw if isinstance(extras_raw, dict) else {}
        merged: dict[str, Any] = {**extras, **raw}
        row = {
            "name": merged.get("name", path.stem),
            "status": merged.get("status"),
            "n_seeds": merged.get("n_seeds"),
            "n_ok": merged.get("n_ok"),
            "steps": merged.get("steps"),
            "sharpe": merged.get("sharpe"),
            "max_drawdown": merged.get("max_drawdown"),
            "n_trades": merged.get("n_trades"),
            "win_rate": merged.get("win_rate"),
            "total_return_pct": merged.get("total_return_pct"),
            "path": str(path),
        }
        for key, val in merged.items():
            if isinstance(key, str) and key.startswith("pnl_dist_") and isinstance(val, (int, float)):
                row[key] = val
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Render a simple fixed-width comparison table."""
    if not rows:
        return "(no ablation reports found)"
    headers = [
        "name",
        "status",
        "sharpe",
        "max_drawdown",
        "n_trades",
        "win_rate",
        "n_ok",
    ]
    # Include any pnl_dist_* keys present.
    dist_keys = sorted({k for r in rows for k in r if str(k).startswith("pnl_dist_")})
    headers.extend(dist_keys)

    def cell(row: dict[str, Any], key: str) -> str:
        val = row.get(key)
        if val is None:
            return "-"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    widths = {h: max(len(h), *(len(cell(r, h)) for r in rows)) for h in headers}
    line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    body = [" | ".join(cell(r, h).ljust(widths[h]) for h in headers) for r in rows]
    return "\n".join([line, sep, *body])


def main() -> None:
    """Print (and optionally save) the ablation comparison table."""
    args = parse_args()
    rows = load_ablation_reports(Path(args.ablations_dir))
    table = format_table(rows)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
