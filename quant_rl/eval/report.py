"""Multi-seed reporting utilities."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .metrics import Metrics, calculate_metrics


def aggregate_seeds(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate a list of per-seed result dicts into a summary DataFrame.

    Each result dict must contain ``equity`` (pd.Series) and optionally
    ``trades`` (pd.DataFrame with ``pnl`` column).
    """
    rows = []
    for i, r in enumerate(results):
        equity = r.get("equity")
        trades = r.get("trades")
        breaches = r.get("breaches", [])
        n_sessions = r.get("n_sessions", 1)

        m = calculate_metrics(
            equity,
            trades=trades,
            n_sessions=n_sessions,
            n_breach_sessions=len(breaches),
        )
        row = {
            "seed": i,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "calmar": m.calmar,
            "max_drawdown": m.max_drawdown,
            "total_return": m.total_return,
            "profit_factor": m.profit_factor,
            "expectancy": m.expectancy,
            "win_rate": m.win_rate,
            "total_trades": m.total_trades,
            "breach_rate": m.breach_rate,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame) -> None:
    print("\n=== Multi-Seed Report ===")
    print(df.describe().round(4).to_string())
    print()


def build_summary_table(m: Metrics) -> str:
    """Return a human-readable text table of a single Metrics instance."""
    lines = [
        "=" * 42,
        f"  {'Metric':<22} {'Value':>12}",
        "-" * 42,
        f"  {'Sharpe':<22} {m.sharpe:>12.4f}",
        f"  {'Sortino':<22} {m.sortino:>12.4f}",
        f"  {'Calmar':<22} {m.calmar:>12.4f}",
        f"  {'Max Drawdown':<22} {m.max_drawdown * 100:>11.2f}%",
        f"  {'Total Return':<22} {m.total_return * 100:>11.2f}%",
        f"  {'Profit Factor':<22} {m.profit_factor:>12.4f}",
        f"  {'Expectancy':<22} {m.expectancy:>12.4f}",
        f"  {'Win Rate':<22} {m.win_rate * 100:>11.2f}%",
        f"  {'Total Trades':<22} {m.total_trades:>12d}",
        f"  {'Total PnL':<22} {m.total_pnl:>12.2f}",
        f"  {'Avg Trade PnL':<22} {m.avg_trade:>12.4f}",
        f"  {'Max Consec Losses':<22} {m.max_consec_loss:>12d}",
        f"  {'Turnover':<22} {m.turnover:>12.6f}",
        f"  {'Breach Rate':<22} {m.breach_rate * 100:>11.2f}%",
        "=" * 42,
    ]
    return "\n".join(lines)


def save_metrics_json(m: Metrics, path: Path | str) -> None:
    """Persist a Metrics instance to a JSON file."""
    Path(path).write_text(json.dumps(asdict(m), indent=2))
