"""Multi-seed reporting utilities."""
from __future__ import annotations

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
