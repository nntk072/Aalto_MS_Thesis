"""Shared Plotly availability, palette, and HTML output."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import plotly.graph_objects as go

    _PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    _PLOTLY_AVAILABLE = False

log = logging.getLogger(__name__)
_TEMPLATE = "plotly_white"
_LONG_COLOR = "#26a69a"
_SHORT_COLOR = "#ef5350"
_CLOSE_COLOR = "#ffd54f"
_EQUITY_COLOR = "#42a5f5"
_PEAK_COLOR = "#90caf9"
_BREACH_COLOR = "#ff1744"

__all__ = [
    "_BREACH_COLOR",
    "_CLOSE_COLOR",
    "_EQUITY_COLOR",
    "_LONG_COLOR",
    "_PEAK_COLOR",
    "_PLOTLY_AVAILABLE",
    "_SHORT_COLOR",
    "_TEMPLATE",
    "_check",
    "_save",
    "go",
    "log",
]


def _check() -> None:
    if not _PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive charts: pip install plotly")


def _save(fig: go.Figure, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
