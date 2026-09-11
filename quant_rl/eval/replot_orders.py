"""Regenerate per-trade order charts (PNG + HTML) for an existing run.

Rebuilds the M1 bars for a saved run's split from its ``config.yaml``
snapshot and re-renders the per-trade order charts directly from the run's
existing ``trades.csv`` — no retraining, no re-running the backtest. Useful
after a charting bug fix (e.g. the trade-pairing fix that dropped
``tp_close`` events) so a historical run's charts can be corrected without
the cost of regenerating the underlying trade log.

Usage
-----
    cd Aalto_MS_Thesis
    uv run python -m quant_rl.eval.replot_orders \\
        --run outputs/20260719_002519_rl_train_seed42 --split testing
    uv run python -m quant_rl.eval.replot_orders \\
        --run outputs/20260719_002519_rl_train_seed42 --split both
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import shutil

import pandas as pd
from omegaconf import OmegaConf

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import get_split_config, split_bars, split_train_test
from quant_rl.eval import plots as _plt
from quant_rl.eval import plots_interactive as _pi
from quant_rl.eval.export import _extract_ftmo_limits, _extract_trade_chart_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_run_config(run_dir: Path) -> Any:
    """Prefer the run's own config snapshot so split boundaries/account
    settings match exactly what produced its trades.csv."""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        log.info("Loading config snapshot: %s", cfg_path)
        return OmegaConf.load(cfg_path)
    log.warning("No config.yaml in %s; falling back to default config", run_dir)
    return load_config([])


def _load_split_bars(cfg: Any, split: str, force: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rebuild primary (and secondary if configured) M1 bars for one split."""
    data = run_pipeline(cfg, force=force)
    primary_m1 = data[cfg.data.primary]["M1"]
    secondary_m1 = data.get(cfg.data.secondary, {}).get("M1")
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, _, _ = split_train_test(primary_m1, primary_m1, train_end, test_start)
    train_sec = test_sec = None
    if secondary_m1 is not None and not secondary_m1.empty:
        train_sec, test_sec = split_bars(secondary_m1, train_end, test_start)
    if split == "training":
        return train_bars, train_sec
    return test_bars, test_sec


def _load_equity(split_dir: Path) -> pd.Series | None:
    path = split_dir / "equity.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = df.columns[0]
    equity = df[col]
    equity.name = "equity"
    return equity


def _replot_split_summary(
    split_dir: Path, cfg: Any, dpi: int, trades: pd.DataFrame | None = None
) -> None:
    """Refresh equity / drawdown / daily-PnL charts from the saved equity.csv."""
    equity = _load_equity(split_dir)
    if equity is None or equity.empty:
        log.warning("No equity.csv in %s; skipping summary charts", split_dir)
        return
    daily_loss_limit, max_loss_limit, profit_target = _extract_ftmo_limits(cfg)
    initial_balance = 100_000.0
    try:
        initial_balance = float(cfg.account.initial_balance)
    except Exception:
        pass
    log.info("Re-plotting equity/drawdown/daily PnL → %s", split_dir)
    _plt.plot_equity_curve(
        equity,
        initial_balance=initial_balance,
        daily_loss_limit=daily_loss_limit,
        max_loss_limit=max_loss_limit,
        profit_target=profit_target,
        trades=trades,
        out_path=split_dir / "equity.png",
        dpi=dpi,
    )
    _plt.plot_drawdown(equity, out_path=split_dir / "drawdown.png", dpi=dpi)
    _plt.plot_daily_pnl(
        equity,
        daily_loss_limit=daily_loss_limit,
        out_path=split_dir / "daily_pnl.png",
        dpi=dpi,
    )
    try:
        _pi.plot_equity_curve(
            equity,
            initial_balance=initial_balance,
            daily_loss_limit=daily_loss_limit,
            max_loss_limit=max_loss_limit,
            profit_target=profit_target,
            trades=trades,
            out_path=split_dir / "equity.html",
        )
        _pi.plot_drawdown(equity, out_path=split_dir / "drawdown.html")
        _pi.plot_daily_pnl(
            equity,
            daily_loss_limit=daily_loss_limit,
            out_path=split_dir / "daily_pnl.html",
        )
    except ImportError:
        log.warning("plotly not available — skipping interactive summary charts")


def _replot_split(run_dir: Path, cfg: Any, split: str, dpi: int, force: bool) -> None:
    split_dir = run_dir / split
    trades = None
    trades_path = split_dir / "trades.csv"
    if trades_path.exists():
        trades = pd.read_csv(trades_path, parse_dates=["time"])
    _replot_split_summary(split_dir, cfg, dpi, trades=trades)
    if trades is None:
        log.warning("No trades.csv for split=%s in %s; skipping", split, run_dir)
        return

    bars, secondary = _load_split_bars(cfg, split, force)

    bar_tz = pd.DatetimeIndex(bars.index).tz
    if bar_tz is not None:
        if trades["time"].dt.tz is None:
            trades["time"] = trades["time"].dt.tz_localize(bar_tz)
        else:
            trades["time"] = trades["time"].dt.tz_convert(bar_tz)

    chart_cfg = _extract_trade_chart_config(cfg)
    orders_dir = split_dir / "orders"

    # Old filenames encode the (possibly wrong) PnL, so a mis-paired trade
    # regenerates under a different filename than its corrected version.
    # Clear stale charts first so the directory doesn't accumulate both.
    if orders_dir.exists():
        shutil.rmtree(orders_dir)

    log.info(
        "Re-plotting PNG orders for split=%s (%d trade rows) → %s", split, len(trades), orders_dir
    )
    _plt.plot_per_trade_orders(
        bars,
        trades,
        orders_dir=orders_dir,
        dpi=dpi,
        secondary_bars=secondary,
        **chart_cfg,
    )

    log.info("Re-plotting HTML orders for split=%s → %s", split, orders_dir)
    _pi.plot_per_trade_orders(
        bars,
        trades,
        orders_dir=orders_dir,
        secondary_bars=secondary,
        **chart_cfg,
    )
    from quant_rl.eval.trade_plots import write_trade_diagnostics

    log.info("Re-plotting trade diagnostics → %s", split_dir)
    write_trade_diagnostics(
        split_dir,
        trades,
        bars,
        dpi=dpi,
        save_plots=True,
        save_html=True,
        lots=float(chart_cfg.get("lots", 1.0)),
        contract_size=float(chart_cfg.get("contract_size", 1.0)),
        max_loss_per_trade_usd=chart_cfg.get("max_loss_per_trade_usd"),
        take_profit_per_trade_usd=chart_cfg.get("take_profit_per_trade_usd"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate order charts and equity/drawdown/daily-PnL figures "
        "for an existing run (no retraining, no backtest re-run)."
    )
    parser.add_argument("--run", required=True, help="Path to an existing run directory")
    parser.add_argument(
        "--split",
        choices=["training", "testing", "both"],
        default="both",
        help="Which split's orders to regenerate",
    )
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    parser.add_argument(
        "--force", action="store_true", help="Force the data pipeline to rebuild caches"
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    cfg = _load_run_config(run_dir)

    splits = ["training", "testing"] if args.split == "both" else [args.split]
    for split in splits:
        _replot_split(run_dir, cfg, split, args.dpi, args.force)

    log.info("Done. Re-plotted orders under: %s", run_dir)


if __name__ == "__main__":
    main()
