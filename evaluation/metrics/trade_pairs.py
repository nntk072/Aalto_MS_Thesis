"""Semantic pairing of trade opens and closes for metric consumers."""

from __future__ import annotations

import pandas as pd

_CLOSE_TYPES = {"close", "forced_close", "eod_close", "stop_close", "tp_close"}


def pair_trades(trades: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    """Return a list of (open_row, close_row) Series pairs for each completed trade.

    Walks the trade log in chronological order (the engine appends rows in
    bar order) with a simple open/close state machine: each ``open`` row is
    paired with the very next row whose type is in ``_CLOSE_TYPES``. This is
    robust to *any* close type — including ``tp_close`` — so a take-profit
    exit can never desync the pairing and cause a later open to be matched
    with the wrong close (which previously mixed one order's entry/volume
    with a different order's PnL on the chart).
    """
    if "bar" in trades.columns and trades["bar"].notna().any():
        trades = trades.sort_values("bar", kind="stable")

    pairs: list[tuple[pd.Series, pd.Series]] = []
    pending_open: pd.Series | None = None
    for _, row in trades.iterrows():
        row_type = row.get("type")
        if row_type == "open":
            pending_open = row
        elif row_type in _CLOSE_TYPES and pending_open is not None:
            pairs.append((pending_open, row))
            pending_open = None
    return pairs


__all__ = ["pair_trades"]
