"""Tabular formatting for decision-diversity metric results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics.decision_diversity import (
    _RR_PCTS,
    DiversityThresholds,
    diversity_report,
)


def _table(
    header: list[str],
    rows: list[list[str]],
    left_cols: tuple[int, ...] = (0,),
) -> list[str]:
    """Align a column table: first column left, the rest right."""
    if not rows:
        return []
    ncol = len(header)
    widths = [len(str(header[i])) for i in range(ncol)]
    for row in rows:
        for i in range(ncol):
            widths[i] = max(widths[i], len(str(row[i])))

    def fmt(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            text = str(cell)
            if i in left_cols:
                parts.append(text.ljust(widths[i]))
            else:
                parts.append(text.rjust(widths[i]))
        return "  ".join(parts).rstrip()

    return [fmt(header)] + [fmt(row) for row in rows]


def format_diversity_table(
    trades: pd.DataFrame,
    thresholds: DiversityThresholds | None = None,
) -> str:
    """Column tables for the trade-decision diversity report.

    Each group renders as its own box with metrics as columns: SL/TP
    summary + families, exit + planned RR, rank + stats, one box per
    conditioner, and warnings.
    """
    from evaluation.reporting.box_table import render_box_table

    report = diversity_report(trades, thresholds)
    opens = report.opens
    boxes: list[tuple[str, list[str]]] = []

    # --- SL / TP summary + family shares ---
    summary_rows: list[list[str]] = []
    for side, shares in (("SL", report.sl_shares), ("TP", report.tp_shares)):
        if shares.empty:
            continue
        probs = shares.to_numpy(dtype=float)
        probs = probs[probs > 0]  # type: ignore[assignment]
        if probs.size:
            entropy = float(-(probs * np.log(probs)).sum())
        else:
            entropy = float("nan")
        summary_rows.append(
            [
                side,
                str(int(shares.shape[0])),
                f"{float(np.exp(entropy)):.2f}",
                f"{entropy:.2f}",
                f"{100.0 * float(shares.max()):.0f}%",
            ]
        )
    if summary_rows:
        block = ["--- summary ---"]
        block += _table(["", "unique", "effective", "entropy", "top"], summary_rows)
        families = sorted(
            set(report.sl_shares.index) | set(report.tp_shares.index),
            key=lambda fam: (
                float(report.sl_shares.get(fam, 0.0)) + float(report.tp_shares.get(fam, 0.0))
            ),
            reverse=True,
        )
        if families:
            family_rows: list[list[str]] = []
            for family in families:
                sl = float(report.sl_shares.get(family, np.nan))
                tp = float(report.tp_shares.get(family, np.nan))
                family_rows.append(
                    [
                        str(family),
                        "-" if not np.isfinite(sl) else f"{100.0 * sl:.0f}%",
                        "-" if not np.isfinite(tp) else f"{100.0 * tp:.0f}%",
                    ]
                )
            block += ["--- families ---"]
            block += _table(["family", "SL", "TP"], family_rows)
        boxes.append(("diversity: sl/tp", block))

    # --- exit modes + planned RR percentiles ---
    block_list: list[str] = []  # renamed to avoid conflict with 'block' parameter
    if not report.exit_shares.empty:
        exit_rows = [
            [str(mode), f"{100.0 * float(share):.0f}%"]
            for mode, share in report.exit_shares.items()
        ]
        block_list += ["--- exit ---"]
        block_list += _table(["mode", "share"], exit_rows)
    if not report.rr.empty:
        rr_vals = report.rr.to_numpy(dtype=float)
        rr_header = [f"P{pct}" for pct in _RR_PCTS]
        rr_row = [f"{float(np.quantile(rr_vals, pct / 100.0)):.2f}" for pct in _RR_PCTS]
        block_list += ["--- planned rr ---"]
        block_list += _table(rr_header, [rr_row], left_cols=())
    if block_list:
        boxes.append(("diversity: exit/rr", block_list))

    # --- rank + extra stats ---
    block = []
    rank_rows: list[list[str]] = []
    if not opens.empty:
        if "sl_rank" in opens.columns:
            arr = pd.to_numeric(opens["sl_rank"], errors="coerce").dropna()
            if not arr.empty:
                values = arr.to_numpy(dtype=float)
                rank_rows.append(
                    [
                        "SL",
                        f"{float(np.median(values)):.2f}",
                        f"{float(np.quantile(values, 0.90)):.2f}",
                    ]
                )
        if "tp_rank" in opens.columns and "exit_mode" in opens.columns:
            mask = opens["exit_mode"].astype(str).ne("ema_21")
            arr = pd.to_numeric(opens.loc[mask, "tp_rank"], errors="coerce").dropna()
            if not arr.empty:
                values = arr.to_numpy(dtype=float)
                rank_rows.append(
                    [
                        "TP",
                        f"{float(np.median(values)):.2f}",
                        f"{float(np.quantile(values, 0.90)):.2f}",
                    ]
                )
    if rank_rows:
        block += ["--- rank ---"]
        block += _table(["", "median", "P90"], rank_rows)
    stat_rows: list[list[str]] = []
    if not opens.empty and "stop_u" in opens.columns:
        stop_u = pd.to_numeric(opens["stop_u"], errors="coerce").dropna()
        if not stop_u.empty:
            stat_rows.append(["stop_u std", f"{float(stop_u.std(ddof=0)):.3f}"])
    if not opens.empty and "sl_family" in opens.columns:
        unique_families = int(opens["sl_family"].replace("", np.nan).nunique(dropna=True))
        stat_rows.append(["unique sl families", str(unique_families)])
    if stat_rows:
        block += ["--- stats ---"]
        block += _table(["stat", "value"], stat_rows)
    if block:
        boxes.append(("diversity: rank/stats", block))

    # --- per-conditioner tables: levels as rows, families as columns ---
    for name, table in report.by_conditioner.items():
        if table.empty:
            continue
        totals = table.sum(axis=1).replace(0, np.nan)
        shares_df = table.astype(float).div(totals, axis=0) * 100.0
        order = list(shares_df.max(axis=0).sort_values(ascending=False).index[:6])
        if not order:
            continue
        header = [name] + [str(family) for family in order]
        body: list[list[str]] = []
        for level, row in shares_df.iterrows():
            cells = [str(level)]
            for family in order:
                value = float(row.get(family, np.nan))
                if not np.isfinite(value) or value <= 0.0:
                    cells.append("-")
                else:
                    cells.append(f"{value:.0f}%")
            body.append(cells)
        boxes.append((f"diversity: by {name}", _table(header, body)))

    # --- warnings ---
    if report.warnings:
        boxes.append(
            (
                "diversity: warnings",
                _table(["warning"], [[str(w)] for w in report.warnings]),
            )
        )

    return render_box_table(boxes)
