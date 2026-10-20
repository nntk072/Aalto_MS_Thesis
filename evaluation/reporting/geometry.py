"""Reporting summaries derived from semantic order-geometry records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def geometry_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(sl_ref, tp_ref)`` from the open records."""
    columns = [
        "sl_ref",
        "tp_ref",
        "n",
        "planned_rr_median",
        "planned_rr_min",
        "planned_rr_max",
        "realized_r_median",
    ]
    if trades.empty or "sl_ref" not in trades.columns:
        return pd.DataFrame(columns=columns)
    opens = trades
    if "type" in trades.columns:
        opens = trades.loc[trades["type"].astype(str).eq("open")]
    if opens.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    grouped = opens.groupby(["sl_ref", "tp_ref"], dropna=False, sort=False)
    for (sl_ref, tp_ref), group in grouped:
        planned = (
            pd.to_numeric(group["planned_rr"], errors="coerce") if "planned_rr" in group else None
        )
        realized = (
            pd.to_numeric(group["realized_r"], errors="coerce") if "realized_r" in group else None
        )
        ema = str(tp_ref) == "ema_21"
        if "exit_mode" in group.columns and group["exit_mode"].astype(str).eq("ema_21").all():
            ema = True
        rows.append(
            {
                "sl_ref": sl_ref,
                "tp_ref": tp_ref,
                "n": int(len(group)),
                "planned_rr_median": np.nan if ema or planned is None else float(planned.median()),
                "planned_rr_min": np.nan if ema or planned is None else float(planned.min()),
                "planned_rr_max": np.nan if ema or planned is None else float(planned.max()),
                "realized_r_median": np.nan if realized is None else float(realized.median()),
            }
        )
    return pd.DataFrame(rows, columns=columns)
