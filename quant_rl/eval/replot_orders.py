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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import shutil

import pandas as pd
from omegaconf import OmegaConf

from quant_rl.config import load_config
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.split import get_split_config, split_train_test
from quant_rl.eval import plots as _plt
from quant_rl.eval import plots_interactive as _pi
from quant_rl.eval.export import _extract_trade_chart_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_run_config(run_dir: Path):
    """Prefer the run's own config snapshot so split boundaries/account
    settings match exactly what produced its trades.csv."""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        log.info("Loading config snapshot: %s", cfg_path)
        return OmegaConf.load(cfg_path)
    log.warning("No config.yaml in %s; falling back to default config", run_dir)
    return load_config([])


def _load_split_bars(cfg, split: str, force: bool) -> pd.DataFrame:
    """Rebuild the raw M1 bars for one split (charts only need OHLC, not features)."""
    data = run_pipeline(cfg, force=force)
    primary_m1 = data[cfg.data.primary]["M1"]
    train_end, test_start = get_split_config(cfg)
    train_bars, test_bars, _, _ = split_train_test(primary_m1, primary_m1, train_end, test_start)
    return train_bars if split == "training" else test_bars


def _replot_split(run_dir: Path, cfg, split: str, dpi: int, force: bool) -> None:
    split_dir = run_dir / split
    trades_path = split_dir / "trades.csv"
    if not trades_path.exists():
        log.warning("No trades.csv for split=%s in %s; skipping", split, run_dir)
        return

    trades = pd.read_csv(trades_path, parse_dates=["time"])
    bars = _load_split_bars(cfg, split, force)

    if bars.index.tz is not None:
        if trades["time"].dt.tz is None:
            trades["time"] = trades["time"].dt.tz_localize(bars.index.tz)
        else:
            trades["time"] = trades["time"].dt.tz_convert(bars.index.tz)

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
    _plt.plot_per_trade_orders(bars, trades, orders_dir=orders_dir, dpi=dpi, **chart_cfg)

    log.info("Re-plotting HTML orders for split=%s → %s", split, orders_dir)
    _pi.plot_per_trade_orders(bars, trades, orders_dir=orders_dir, **chart_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate per-trade order charts for an existing run's trades.csv "
        "(no retraining, no backtest re-run)."
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
