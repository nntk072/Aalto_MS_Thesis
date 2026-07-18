"""Artifact export: save a complete run directory.

New layout (v2)
---------------
::

    outputs/<ts>_<name>/
        config.yaml          — OmegaConf config snapshot
        summary.txt          — Train vs Test comparison table
        training/            — in-sample  (≤ train_end)
            equity.csv, trades.csv, metrics.json
            breach_events.csv, session_activity.json
            equity.png/html, drawdown.png/html
            pnl_hist.png/html, returns_dist.png, monthly_heatmap.png/html
            orders/          — one M1 candlestick per trade
                trade_NNNN_YYYYMMDD_HHMMopen_HHMMclose_{L|S}_{p|m}PnL.png
                trade_NNNN_…  .html
        testing/             — out-of-sample (≥ test_start)
            (same set)
        model/               — RL runs only (populated by train_rl.py)
            ppo_*.zip, training_log.csv
            learning_curve.png/html, losses.png/html …

Usage::

    from quant_rl.eval.export import save_run, build_run_dir
    run_dir = save_run(
        out_dir="outputs", name="random",
        train_result=train_result, train_metrics=train_m, train_bars=train_bars,
        test_result=test_result,  test_metrics=test_m,  test_bars=test_bars,
        cfg=cfg,
    )
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import Metrics
from .report import build_summary_table, build_comparison_table, save_metrics_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_run_dir(base: str | Path, name: str) -> Path:
    """Return a new timestamped directory path (does not create it yet)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"{ts}_{name}"


def _extract_ftmo_limits(cfg: Any) -> tuple[float | None, float | None]:
    daily_loss_limit = max_loss_limit = None
    if cfg is not None:
        try:
            daily_loss_limit = cfg.ftmo.daily_loss_limit
            max_loss_limit   = cfg.ftmo.max_loss_limit
        except Exception:
            pass
    return daily_loss_limit, max_loss_limit


# ---------------------------------------------------------------------------
# Per-split writer (all data + charts for one split go into one subfolder)
# ---------------------------------------------------------------------------

def _write_split(
    split_dir: Path,
    result: dict[str, Any],
    metrics: Metrics,
    bars: pd.DataFrame | None,
    daily_loss_limit: float | None,
    max_loss_limit: float | None,
    save_plots: bool,
    save_html: bool,
    save_csv: bool,
    dpi: int,
    max_order_charts: int,
) -> None:
    """Write all data + charts for one split into *split_dir*."""
    split_dir.mkdir(parents=True, exist_ok=True)

    equity: pd.Series         = result.get("equity", pd.Series(dtype=float))
    trades: pd.DataFrame      = result.get("trades", pd.DataFrame())
    breach_events: list[dict] = result.get("breach_events", [])
    initial_balance: float    = result.get("initial_balance", 100_000.0)

    # ------------------------------------------------------------------
    # CSVs + diagnostics
    # ------------------------------------------------------------------
    if save_csv:
        if not equity.empty:
            equity.to_csv(split_dir / "equity.csv", header=["equity"])
        if not trades.empty:
            trades.to_csv(split_dir / "trades.csv", index=False)
        if breach_events:
            pd.DataFrame(breach_events).to_csv(split_dir / "breach_events.csv", index=False)
        session_diag = {
            "n_sessions":             result.get("n_sessions", 0),
            "n_breach_sessions":      result.get("n_breach_sessions", 0),
            "n_sessions_with_trades": result.get("n_sessions_with_trades", 0),
            "n_sessions_skipped":     result.get("n_sessions_skipped", 0),
        }
        (split_dir / "session_activity.json").write_text(json.dumps(session_diag, indent=2))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    save_metrics_json(metrics, split_dir / "metrics.json")
    (split_dir / "summary.txt").write_text(build_summary_table(metrics))

    # ------------------------------------------------------------------
    # Static PNG charts
    # ------------------------------------------------------------------
    if save_plots and not equity.empty:
        from . import plots as _plt

        _plt.plot_equity_curve(
            equity, breach_events=breach_events,
            initial_balance=initial_balance,
            daily_loss_limit=daily_loss_limit,
            max_loss_limit=max_loss_limit,
            out_path=split_dir / "equity.png", dpi=dpi,
        )
        _plt.plot_drawdown(equity, out_path=split_dir / "drawdown.png", dpi=dpi)

        if not trades.empty:
            _plt.plot_trade_pnl_hist(trades, out_path=split_dir / "pnl_hist.png", dpi=dpi)

        _plt.plot_returns_dist(equity, out_path=split_dir / "returns_dist.png", dpi=dpi)
        _plt.plot_monthly_returns_heatmap(
            equity, out_path=split_dir / "monthly_heatmap.png", dpi=dpi
        )

        if bars is not None and not trades.empty:
            _plt.plot_per_trade_orders(
                bars, trades,
                orders_dir=split_dir / "orders",
                max_charts=max_order_charts,
                dpi=dpi,
            )

    # ------------------------------------------------------------------
    # Interactive HTML charts
    # ------------------------------------------------------------------
    if save_html and not equity.empty:
        try:
            from . import plots_interactive as _pi

            _pi.plot_equity_curve(
                equity, breach_events=breach_events,
                initial_balance=initial_balance,
                daily_loss_limit=daily_loss_limit,
                max_loss_limit=max_loss_limit,
                out_path=split_dir / "equity.html",
            )
            _pi.plot_drawdown(equity, out_path=split_dir / "drawdown.html")

            if not trades.empty:
                _pi.plot_trade_pnl_hist(trades, out_path=split_dir / "pnl_hist.html")

            _pi.plot_returns_dist(equity, out_path=split_dir / "returns_hist.html")
            _pi.plot_monthly_returns_heatmap(equity, out_path=split_dir / "monthly_heatmap.html")

            if bars is not None and not trades.empty:
                _pi.plot_per_trade_orders(
                    bars, trades,
                    orders_dir=split_dir / "orders",
                    max_charts=max_order_charts,
                )
        except ImportError:
            log.warning("plotly not available — skipping interactive HTML charts")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_run(
    out_dir: str | Path = "outputs",
    name: str = "run",
    run_dir: Path | None = None,
    train_result: dict[str, Any] | None = None,
    train_metrics: Metrics | None = None,
    test_result: dict[str, Any] | None = None,
    test_metrics: Metrics | None = None,
    train_bars: pd.DataFrame | None = None,
    test_bars: pd.DataFrame | None = None,
    cfg: Any | None = None,
    save_plots: bool = True,
    save_html: bool = True,
    save_csv: bool = True,
    dpi: int = 150,
    max_order_charts: int = 200,
) -> Path:
    """Persist a complete train + test run.

    Parameters
    ----------
    out_dir, name:
        Base directory and run label.  Creates ``<out_dir>/<ts>_<name>/``.
    run_dir:
        Explicit pre-created directory (overrides *out_dir* / *name* if set).
        Useful when the caller has already written model weights into the same
        directory (e.g. ``train_rl.py``).
    train_result, train_metrics, train_bars:
        In-sample backtest result dict, metrics, and price bars.
    test_result, test_metrics, test_bars:
        Out-of-sample equivalents.  Either split may be ``None``.

    Returns
    -------
    Path to the run directory.
    """
    if run_dir is None:
        run_dir = build_run_dir(out_dir, name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving run artifacts to %s", run_dir)

    daily_loss_limit, max_loss_limit = _extract_ftmo_limits(cfg)

    # Config snapshot at root only
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

    split_kwargs: dict[str, Any] = dict(
        daily_loss_limit=daily_loss_limit,
        max_loss_limit=max_loss_limit,
        save_plots=save_plots,
        save_html=save_html,
        save_csv=save_csv,
        dpi=dpi,
        max_order_charts=max_order_charts,
    )

    if train_result is not None and train_metrics is not None:
        _write_split(run_dir / "training", train_result, train_metrics,
                     train_bars, **split_kwargs)

    if test_result is not None and test_metrics is not None:
        _write_split(run_dir / "testing", test_result, test_metrics,
                     test_bars, **split_kwargs)

    # Comparison summary at root
    if train_metrics is not None:
        (run_dir / "summary.txt").write_text(
            build_comparison_table(train_metrics, test_metrics)
        )

    log.info("Run saved: %s", run_dir)
    return run_dir

