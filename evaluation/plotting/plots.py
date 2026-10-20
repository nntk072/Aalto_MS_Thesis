"""Compatibility API for static plots, grouped by chart family."""

from evaluation.metrics.trade_pairs import pair_trades as _pair_trades

from .common import (
    BREACH_COLOR,
    CLOSE_COLOR,
    EQUITY_COLOR,
    LONG_COLOR,
    PEAK_COLOR,
    SHORT_COLOR,
    STYLE,
    _apply_style,
    _save,
)
from .comparison import plot_baseline_comparison
from .equity import _draw_dd_panel, plot_drawdown, plot_equity_curve
from .order_window import _extend_trade_window, _extract_window, _structure_anchors
from .orders import (
    _axis_date_format,
    _candle_body_days,
    _close_time_label,
    _per_trade_equity,
    _plot_price_line_fallback,
    _trade_filename,
    plot_per_trade_orders,
    plot_price_with_orders,
)
from .performance import (
    plot_daily_pnl,
    plot_direction_summary,
    plot_monthly_returns_heatmap,
    plot_returns_dist,
    plot_trade_pnl_hist,
)

__all__ = [
    "BREACH_COLOR",
    "CLOSE_COLOR",
    "EQUITY_COLOR",
    "LONG_COLOR",
    "PEAK_COLOR",
    "SHORT_COLOR",
    "STYLE",
    "_apply_style",
    "_axis_date_format",
    "_candle_body_days",
    "_close_time_label",
    "_draw_dd_panel",
    "_extend_trade_window",
    "_extract_window",
    "_pair_trades",
    "_per_trade_equity",
    "_plot_price_line_fallback",
    "_save",
    "_structure_anchors",
    "_trade_filename",
    "plot_baseline_comparison",
    "plot_daily_pnl",
    "plot_direction_summary",
    "plot_drawdown",
    "plot_equity_curve",
    "plot_monthly_returns_heatmap",
    "plot_per_trade_orders",
    "plot_price_with_orders",
    "plot_returns_dist",
    "plot_trade_pnl_hist",
]
