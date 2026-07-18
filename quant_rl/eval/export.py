"""Artifact export: save a complete run directory with all charts, CSVs, and JSON.

Usage (programmatic)::

    from quant_rl.eval.export import save_run
    save_run(result, metrics, out_dir="outputs", name="random", bars=primary_m1, cfg=cfg)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import Metrics
from .report import build_summary_table, save_metrics_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_run_dir(base: str | Path, name: str) -> Path:
    """Return a *new* timestamped directory path (does not create it yet)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"{ts}_{name}"


def save_run(
    result: dict[str, Any],
    metrics: Metrics,
    out_dir: str | Path = "outputs",
    name: str = "run",
    bars: pd.DataFrame | None = None,
    cfg: Any | None = None,
    save_plots: bool = True,
    save_html: bool = True,
    save_csv: bool = True,
    dpi: int = 150,
    candle_tf: str = "15min",
    price_plot_max_points: int = 3000,
) -> Path:
    """Persist a complete backtest/eval run.

    Creates ``<out_dir>/<timestamp>_<name>/`` and writes:

    * ``equity.csv``            — equity curve
    * ``trades.csv``            — trade log
    * ``metrics.json``          — all metrics
    * ``summary.txt``           — human-readable table
    * ``config.yaml``           — OmegaConf config snapshot (if *cfg* provided)
    * ``equity.png``            — equity + drawdown PNG
    * ``drawdown.png``
    * ``orders.png``            — candlestick + order markers (if *bars* provided)
    * ``pnl_hist.png``
    * ``returns_dist.png``
    * ``monthly_heatmap.png``
    * ``equity.html``           — interactive equity
    * ``drawdown.html``
    * ``orders.html``           — interactive candlestick
    * ``pnl_hist.html``
    * ``returns_hist.html``
    * ``monthly_heatmap.html``

    Returns the run directory path.
    """
    run_dir = build_run_dir(out_dir, name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving run artifacts to %s", run_dir)

    equity: pd.Series = result.get("equity", pd.Series(dtype=float))
    trades: pd.DataFrame = result.get("trades", pd.DataFrame())
    breaches: list[str] = result.get("breaches", [])
    initial_balance: float = result.get("initial_balance", 100_000.0)

    # ------------------------------------------------------------------
    # CSVs
    # ------------------------------------------------------------------
    if save_csv:
        if not equity.empty:
            equity.to_csv(run_dir / "equity.csv", header=["equity"])
        if not trades.empty:
            trades.to_csv(run_dir / "trades.csv", index=False)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    save_metrics_json(metrics, run_dir / "metrics.json")
    (run_dir / "summary.txt").write_text(build_summary_table(metrics))

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Extract FTMO limits from cfg if available
    # ------------------------------------------------------------------
    daily_loss_limit = max_loss_limit = None
    if cfg is not None:
        try:
            daily_loss_limit = cfg.ftmo.daily_loss_limit
            max_loss_limit   = cfg.ftmo.max_loss_limit
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Static PNG charts
    # ------------------------------------------------------------------
    if save_plots and not equity.empty:
        from . import plots as _plt

        _plt.plot_equity_curve(
            equity, breaches=breaches,
            initial_balance=initial_balance,
            daily_loss_limit=daily_loss_limit,
            max_loss_limit=max_loss_limit,
            out_path=run_dir / "equity.png", dpi=dpi,
        )
        _plt.plot_drawdown(equity, out_path=run_dir / "drawdown.png", dpi=dpi)

        if not trades.empty:
            _plt.plot_trade_pnl_hist(trades, out_path=run_dir / "pnl_hist.png", dpi=dpi)

        _plt.plot_returns_dist(equity, out_path=run_dir / "returns_dist.png", dpi=dpi)
        _plt.plot_monthly_returns_heatmap(equity, out_path=run_dir / "monthly_heatmap.png", dpi=dpi)

        if bars is not None and not trades.empty:
            _plt.plot_price_with_orders(
                bars, trades,
                max_points=price_plot_max_points,
                candle_tf=candle_tf,
                out_path=run_dir / "orders.png", dpi=dpi,
            )

    # ------------------------------------------------------------------
    # Interactive HTML charts
    # ------------------------------------------------------------------
    if save_html and not equity.empty:
        try:
            from . import plots_interactive as _pi

            _pi.plot_equity_curve(
                equity, breaches=breaches,
                initial_balance=initial_balance,
                daily_loss_limit=daily_loss_limit,
                max_loss_limit=max_loss_limit,
                out_path=run_dir / "equity.html",
            )
            _pi.plot_drawdown(equity, out_path=run_dir / "drawdown.html")

            if not trades.empty:
                _pi.plot_trade_pnl_hist(trades, out_path=run_dir / "pnl_hist.html")

            _pi.plot_returns_dist(equity, out_path=run_dir / "returns_hist.html")
            _pi.plot_monthly_returns_heatmap(equity, out_path=run_dir / "monthly_heatmap.html")

            if bars is not None and not trades.empty:
                _pi.plot_price_with_orders(
                    bars, trades,
                    max_points=price_plot_max_points,
                    candle_tf=candle_tf,
                    out_path=run_dir / "orders.html",
                )
        except ImportError:
            log.warning("plotly not available — skipping interactive HTML charts")

    log.info("Run saved: %s", run_dir)
    return run_dir
