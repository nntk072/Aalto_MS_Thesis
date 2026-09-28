"""Decoded trade-decision diversity.

Raw action spread and the selected market reference are different objects.
This module reads open rows only. It does not change the reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_RR_PCTS: tuple[int, ...] = (10, 25, 50, 75, 90, 95)
_CONDITIONERS: tuple[str, ...] = (
    "strategy",
    "session",
    "manipulation",
    "volatility_regime",
    "trend_regime",
    "direction",
)


@dataclass(frozen=True)
class DiversityThresholds:
    """Printed warnings. None of these enter the reward."""

    sl_top_share: float = 0.80
    tp_top_share: float = 0.80
    rr_p75: float = 1.25
    ema_lo: float = 0.02
    ema_hi: float = 0.98
    action_std: float = 0.05


def thresholds_from_mapping(raw: dict[str, Any] | None) -> DiversityThresholds:
    """Build thresholds from ``training.dashboard.diversity``."""
    base = DiversityThresholds()
    if not raw:
        return base
    values = {field: raw[field] for field in base.__dataclass_fields__ if field in raw}
    return DiversityThresholds(**values)


def reference_family(name: object) -> str:
    """Coarse family of a level name. A joined name uses its first piece.

    An ``M5_`` / ``M15_`` / ``H1_`` prefix is ``htf``, so a higher-timeframe
    swing is not counted as the M1 swing.
    """
    text = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name)
    head = text.split("|", 1)[0].strip().lower()
    if not head or head == "ema_21":
        return ""
    for prefix in ("h1_", "m15_", "m5_"):
        if head.startswith(prefix):
            return "htf"
    if "sweep" in head:
        return "sweep"
    if "fvg" in head:
        return "fvg"
    if head.startswith("dev_") or head == "dev":
        return "deviation"
    if "prev_week" in head:
        return "prev_week"
    if "prev_day" in head:
        return "prev_day"
    if head.startswith("asian") or head.startswith("london"):
        return "session"
    if "swing" in head:
        return "swing"
    return head


def candidate_rank(index: object, count: object) -> float:
    """Ordinal place of a choice in ``[0, 1]``. One candidate has rank 0.

    A missing index, including an EMA exit, is NaN.
    """
    try:
        idx = float(index)  # type: ignore[arg-type]
        n = float(count)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(idx) or not np.isfinite(n) or idx < 0.0 or n <= 0.0:
        return float("nan")
    if n <= 1.0:
        return 0.0
    return float(idx) / (n - 1.0)


def shannon_entropy(labels: pd.Series) -> float:
    """Entropy in nats. A single label has entropy 0."""
    clean = labels.astype(str)
    clean = clean[clean.ne("") & clean.ne("nan")]
    if clean.empty:
        return float("nan")
    prob = clean.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-np.sum(prob * np.log(prob)))


@dataclass
class DiversityReport:
    """Shares, quantiles, and the terminal text for one set of opens."""

    text: str
    warnings: list[str]
    sl_shares: pd.Series
    tp_shares: pd.Series
    exit_shares: pd.Series
    joint: pd.DataFrame
    rr: pd.Series
    opens: pd.DataFrame
    by_conditioner: dict[str, pd.DataFrame]


def diversity_report(
    trades: pd.DataFrame,
    thresholds: DiversityThresholds | None = None,
) -> DiversityReport:
    """Summarize open rows. Empty input still returns a report."""
    limits = thresholds or DiversityThresholds()
    opens = _opens(trades)
    if opens.empty:
        text = "TRADE DECISION DIVERSITY\nno strategy opens\n"
        return DiversityReport(
            text,
            [],
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            pd.Series(dtype=float),
            opens,
            {},
        )
    frame = opens.copy()
    frame["sl_family"] = _label(frame, "sl_ref").map(reference_family)
    frame["tp_family"] = _label(frame, "tp_ref").map(reference_family)
    frame["sl_rank"] = _ranks(frame, "sl_index", "n_sl")
    frame["tp_rank"] = _ranks(frame, "tp_index", "n_tp")
    if "exit_mode" in frame.columns:
        structural = frame["exit_mode"].astype(str).ne("ema_21")
    else:
        structural = pd.Series(True, index=frame.index)
    tp_labels = frame.loc[structural, "tp_family"]
    sl_shares = _shares(frame["sl_family"])
    tp_shares = _shares(tp_labels)
    exit_shares = _exit_shares(frame)
    joint = _joint(frame.loc[structural])
    if "planned_rr" in frame.columns:
        rr = pd.to_numeric(frame.loc[structural, "planned_rr"], errors="coerce")
    else:
        rr = pd.Series(dtype=float)
    rr = rr.dropna()
    grouped = {
        name: _conditional_counts(frame, name) for name in _CONDITIONERS if name in frame.columns
    }
    warnings = _warnings(frame, sl_shares, tp_shares, exit_shares, rr, limits)
    text = _format(frame, sl_shares, tp_shares, exit_shares, rr, warnings, grouped)
    return DiversityReport(
        text, warnings, sl_shares, tp_shares, exit_shares, joint, rr, frame, grouped
    )


def format_diversity_text(
    trades: pd.DataFrame,
    thresholds: DiversityThresholds | None = None,
) -> str:
    """Terminal block for one trade log."""
    return diversity_report(trades, thresholds).text


def write_decision_artifacts(
    trades: pd.DataFrame,
    directory: Path,
    *,
    thresholds: DiversityThresholds | None = None,
    save_csv: bool = True,
    save_plots: bool = True,
    dpi: int = 100,
) -> DiversityReport:
    """Write the text report, count tables, and the diagnostic figures."""
    directory.mkdir(parents=True, exist_ok=True)
    report = diversity_report(trades, thresholds)
    if save_csv:
        (directory / "decision_diversity.txt").write_text(report.text)
        _write_shares(directory / "sl_family_counts.csv", report.sl_shares)
        _write_shares(directory / "tp_family_counts.csv", report.tp_shares)
        if not report.joint.empty:
            report.joint.to_csv(directory / "sl_tp_joint.csv")
        for name, table in report.by_conditioner.items():
            if not table.empty:
                table.to_csv(directory / f"sl_by_{name}.csv")
    if save_plots and not report.opens.empty:
        _write_figures(report, directory, dpi)
    return report


def _label(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index)
    return frame[column]


def _ranks(frame: pd.DataFrame, index_name: str, count_name: str) -> list[float]:
    idx = frame[index_name] if index_name in frame.columns else pd.Series(np.nan, index=frame.index)
    count = (
        frame[count_name] if count_name in frame.columns else pd.Series(np.nan, index=frame.index)
    )
    return [candidate_rank(a, b) for a, b in zip(idx.tolist(), count.tolist(), strict=False)]


def _opens(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "sl_ref" not in trades.columns:
        return pd.DataFrame()
    if "type" in trades.columns:
        return trades.loc[trades["type"].astype(str).eq("open")].copy()
    return trades.copy()


def _shares(labels: pd.Series) -> pd.Series:
    clean = labels.astype(str)
    clean = clean[clean.ne("") & clean.ne("nan")]
    if clean.empty:
        return pd.Series(dtype=float)
    return clean.value_counts(normalize=True).sort_values(ascending=False)


def _exit_shares(frame: pd.DataFrame) -> pd.Series:
    if "exit_mode" not in frame.columns:
        return pd.Series(dtype=float)
    labels = frame["exit_mode"].astype(str)
    labels = labels[labels.ne("") & labels.ne("nan")]
    if labels.empty:
        return pd.Series(dtype=float)
    return labels.value_counts(normalize=True).sort_values(ascending=False)


def _joint(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    sl = frame["sl_family"].astype(str)
    tp = frame["tp_family"].astype(str)
    keep = sl.ne("") & tp.ne("") & sl.ne("nan") & tp.ne("nan")
    if not bool(keep.any()):
        return pd.DataFrame()
    return pd.crosstab(sl[keep], tp[keep])


def _conditional_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    group = frame[column]
    if group is None:
        return pd.DataFrame()
    labels = group.astype(str)
    keep = labels.ne("") & labels.ne("nan") & frame["sl_family"].astype(str).ne("")
    if not bool(keep.any()):
        return pd.DataFrame()
    return pd.crosstab(labels[keep], frame.loc[keep, "sl_family"])


def _warnings(
    frame: pd.DataFrame,
    sl_shares: pd.Series,
    tp_shares: pd.Series,
    exit_shares: pd.Series,
    rr: pd.Series,
    limits: DiversityThresholds,
) -> list[str]:
    found: list[str] = []
    if not sl_shares.empty and float(sl_shares.iloc[0]) > limits.sl_top_share:
        found.append(
            f"SL COLLAPSE  top {sl_shares.index[0]} {100.0 * float(sl_shares.iloc[0]):.0f}%"
        )
    if not tp_shares.empty and float(tp_shares.iloc[0]) > limits.tp_top_share:
        found.append(
            f"TP COLLAPSE  top {tp_shares.index[0]} {100.0 * float(tp_shares.iloc[0]):.0f}%"
        )
    if not rr.empty:
        p75 = float(np.quantile(rr.to_numpy(dtype=float), 0.75))
        if p75 < limits.rr_p75:
            found.append(f"RR COLLAPSE  P75 {p75:.2f}")
    if not exit_shares.empty and "ema_21" in exit_shares.index:
        ema = float(exit_shares["ema_21"])
        if ema < limits.ema_lo or ema > limits.ema_hi:
            found.append(f"EXIT COLLAPSE  ema_21 {100.0 * ema:.1f}%")
    elif not exit_shares.empty and float(exit_shares.iloc[0]) > limits.ema_hi:
        found.append(
            f"EXIT COLLAPSE  {exit_shares.index[0]} {100.0 * float(exit_shares.iloc[0]):.1f}%"
        )
    for column, label in (("stop_u", "stop_u"), ("target_u", "target_u")):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.size >= 2 and float(values.std(ddof=0)) < limits.action_std:
            found.append(f"ACTION COLLAPSE  {label} std {float(values.std(ddof=0)):.3f}")
    return found


def _format(
    frame: pd.DataFrame,
    sl_shares: pd.Series,
    tp_shares: pd.Series,
    exit_shares: pd.Series,
    rr: pd.Series,
    warnings: list[str],
    grouped: dict[str, pd.DataFrame],
) -> str:
    lines = ["TRADE DECISION DIVERSITY", _block("SL", frame["sl_family"], sl_shares)]
    if "exit_mode" in frame.columns:
        structural = frame["exit_mode"].astype(str).ne("ema_21")
    else:
        structural = pd.Series(True, index=frame.index)
    lines.append(_block("TP", frame.loc[structural, "tp_family"], tp_shares))
    exit_line = "Exit:"
    if exit_shares.empty:
        exit_line += "  n/a"
    else:
        exit_line += "  " + "  ".join(
            f"{name} {100.0 * float(share):.0f}%" for name, share in exit_shares.items()
        )
    lines.append(exit_line)
    lines.append("Planned RR:  " + _rr_text(rr))
    lines.append("SL rank:  " + _rank_text(frame["sl_rank"]))
    lines.append("TP rank:  " + _rank_text(frame.loc[structural, "tp_rank"]))
    if "stop_u" in frame.columns:
        stop_u = pd.to_numeric(frame["stop_u"], errors="coerce")
        lines.append(
            f"stop_u std {float(stop_u.std(ddof=0)) if stop_u.notna().sum() else float('nan'):.3f}"
            f"  unique sl families {int(frame['sl_family'].replace('', np.nan).nunique(dropna=True))}"
        )
    for name, table in grouped.items():
        if table.empty or len(table.index) < 1:
            continue
        lines.append(f"by {name}:")
        for level, row in table.iterrows():
            total = float(row.sum())
            if total <= 0.0:
                continue
            top = row.sort_values(ascending=False).head(4)
            bits = "  ".join(
                f"{col} {100.0 * float(value) / total:.0f}%" for col, value in top.items() if value
            )
            lines.append(f"  {level}  {bits}")
    for warning in warnings:
        lines.append(f"WARNING  {warning}")
    return "\n".join(lines) + "\n"


def _block(title: str, labels: pd.Series, shares: pd.Series) -> str:
    clean = labels.astype(str)
    clean = clean[clean.ne("") & clean.ne("nan")]
    entropy = shannon_entropy(clean)
    effective = float(np.exp(entropy)) if np.isfinite(entropy) else float("nan")
    top = float(shares.iloc[0]) if not shares.empty else float("nan")
    listed = "  ".join(f"{name} {100.0 * float(share):.0f}%" for name, share in shares.items())
    return (
        f"{title}:  unique {clean.nunique()}  effective {effective:.2f}  "
        f"entropy {entropy:.2f}  top {100.0 * top:.0f}%  {listed}"
    )


def _rr_text(rr: pd.Series) -> str:
    if rr.empty:
        return "n/a"
    values = rr.to_numpy(dtype=float)
    parts = [f"P{pct} {float(np.quantile(values, pct / 100.0)):.2f}" for pct in _RR_PCTS]
    return "  ".join(parts)


def _rank_text(ranks: pd.Series) -> str:
    values = pd.to_numeric(ranks, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    arr = values.to_numpy(dtype=float)
    return f"median {float(np.median(arr)):.2f}  P90 {float(np.quantile(arr, 0.90)):.2f}"


def _write_shares(path: Path, shares: pd.Series) -> None:
    if shares.empty:
        pd.DataFrame(columns=["family", "share"]).to_csv(path, index=False)
        return
    shares.rename("share").rename_axis("family").reset_index().to_csv(path, index=False)


def _write_figures(report: DiversityReport, directory: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _bar(report.sl_shares, directory / "sl_family.png", "SL family", dpi)
    _bar(report.tp_shares, directory / "tp_family.png", "TP family", dpi)
    _rr_hist(report.rr, directory / "planned_rr_hist.png", dpi)
    _heatmap(report.opens, directory / "sl_tp_rank_heatmap.png", dpi)
    _manipulation(
        report.by_conditioner.get("manipulation", pd.DataFrame()),
        directory / "sl_by_manipulation.png",
        dpi,
    )
    _staircase(report.opens, "stop_u", "sl_index", directory / "stop_u_vs_index.png", dpi)
    _staircase(report.opens, "target_u", "tp_index", directory / "target_u_vs_index.png", dpi)
    plt.close("all")


def _bar(shares: pd.Series, path: Path, title: str, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    if shares.empty:
        ax.set_title(title)
    else:
        ax.bar(list(shares.index.astype(str)), shares.to_numpy(dtype=float))
        ax.set_ylabel("share")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _rr_hist(rr: pd.Series, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("Planned RR")
    if not rr.empty:
        values = rr.to_numpy(dtype=float)
        ax.hist(values, bins=min(20, max(5, values.size // 2)))
        for pct in _RR_PCTS:
            ax.axvline(float(np.quantile(values, pct / 100.0)), color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _heatmap(opens: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title("SL rank x TP rank")
    sl = pd.to_numeric(opens["sl_rank"], errors="coerce")
    tp = pd.to_numeric(opens["tp_rank"], errors="coerce")
    pair = pd.DataFrame({"sl": sl, "tp": tp}).dropna()
    edges = np.linspace(0.0, 1.0, 6)
    if not pair.empty:
        hist, _, _ = np.histogram2d(pair["sl"], pair["tp"], bins=edges)
        ax.imshow(hist.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlabel("SL rank")
    ax.set_ylabel("TP rank")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _manipulation(table: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_title("SL family by manipulation")
    if not table.empty:
        shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        shares.plot(kind="bar", stacked=True, ax=ax, legend=True)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _staircase(opens: pd.DataFrame, x_name: str, y_name: str, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title(f"{x_name} to {y_name}")
    if x_name in opens.columns and y_name in opens.columns:
        x = pd.to_numeric(opens[x_name], errors="coerce")
        y = pd.to_numeric(opens[y_name], errors="coerce")
        keep = x.notna() & y.notna()
        ax.scatter(x[keep], y[keep], s=12)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
