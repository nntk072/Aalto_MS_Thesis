from __future__ import annotations

from pathlib import Path

import pandas as pd

from .interactive_common import (
    _CLOSE_COLOR,
    _EQUITY_COLOR,
    _LONG_COLOR,
    _SHORT_COLOR,
    _TEMPLATE,
    _check,
    _save,
    go,
)


def plot_baseline_comparison(
    equity_dict: dict[str, pd.Series],
    out_path: Path | str | None = None,
) -> go.Figure:
    _check()
    colors = [_EQUITY_COLOR, _LONG_COLOR, _SHORT_COLOR, _CLOSE_COLOR, "#ce93d8", "#80cbc4"]

    fig = go.Figure()
    for (name, eq), col in zip(equity_dict.items(), colors):
        normed = (eq / eq.iloc[0] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=normed.index,
                y=normed.values,
                mode="lines",
                name=name,
                line=dict(color=col, width=1.5),
            )
        )

    fig.add_hline(y=0, line_color="#555", line_dash="dot")
    fig.update_layout(
        template=_TEMPLATE,
        title="Strategy Comparison — Normalized Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        height=450,
    )

    if out_path:
        _save(fig, out_path)
    return fig
