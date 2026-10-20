"""Shared Matplotlib setup, palette, and file output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

STYLE = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
    "grid.color": "#e6e6e6",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "font.size": 10,
}

LONG_COLOR = "#26a69a"

SHORT_COLOR = "#ef5350"

_NOTE_FONT_PT = 8

CLOSE_COLOR = "#ffd54f"

EQUITY_COLOR = "#42a5f5"

PEAK_COLOR = "#90caf9"

BREACH_COLOR = "#ff1744"

_CLOSE_TYPES = {"close", "forced_close", "eod_close", "stop_close", "tp_close"}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: Figure, path: Path | str, dpi: int = 150) -> Figure:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig
