from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.figure import Figure

from .common import (
    CLOSE_COLOR,
    EQUITY_COLOR,
    LONG_COLOR,
    SHORT_COLOR,
    _apply_style,
    _save,
)
from .order_window import _extend_trade_window as _extend_trade_window
from .order_window import _extract_window as _extract_window
from .order_window import _structure_anchors as _structure_anchors


def plot_baseline_comparison(
    equity_dict: dict[str, pd.Series],
    out_path: Path | str | None = None,
    dpi: int = 150,
) -> Figure:
    """Overlay normalized equity curves for multiple strategies."""
    _apply_style()
    colors = [EQUITY_COLOR, LONG_COLOR, SHORT_COLOR, CLOSE_COLOR, "#ce93d8", "#80cbc4"]

    fig, ax = plt.subplots(figsize=(14, 5))
    for (name, eq), col in zip(equity_dict.items(), colors):
        normed = eq / eq.iloc[0]
        ax.plot(normed.index, normed.values, label=name, color=col)

    ax.axhline(1.0, color="#555", linewidth=0.7, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{(x - 1) * 100:.0f}%"))
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    ax.set_title("Strategy Comparison — Normalized Returns", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)
    fig.tight_layout()

    if out_path:
        _save(fig, out_path, dpi)
    return fig
