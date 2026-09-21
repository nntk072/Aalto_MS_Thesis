"""Build the thesis defense PNG catalog (no PowerPoint / PDF).

Usage
-----
    python scripts/defense_figures.py
    python scripts/defense_figures.py --run-dir outputs/20260920_232943_rl_train_seed50
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import DictConfig, OmegaConf

from quant_rl.config import load_config
from quant_rl.data.activity import attach_activity, tick_counts_from_index
from quant_rl.data.pipeline import run_pipeline
from quant_rl.data.session import ny_session_mask
from quant_rl.data.split import get_split_config, make_train_mask
from quant_rl.eval.data_plots import plot_split_overlay
from quant_rl.eval.defense_plots import (
    plot_activity_24h,
    plot_coverage_with_gaps,
    plot_curated_feature_hists,
    plot_event_rates,
    plot_ifvg_event_study,
    plot_ifvg_occupancy_monthly,
    plot_ny_return_tails,
    plot_pca_ifvg,
    plot_spread_regime,
)
from quant_rl.eval.plots import plot_daily_pnl, plot_drawdown, plot_equity_curve
from quant_rl.eval.po3_plots import plot_annotated_ny_session, plot_fvg_signals
from quant_rl.features.build import build_features, feature_cache_path

log = logging.getLogger(__name__)

_SEED50 = Path("outputs/20260920_232943_rl_train_seed50")
_COPY = (
    ("data/train_test_split.png", "train_test_split.png"),
    ("training/equity.png", "train_equity.png"),
    ("training/daily_pnl.png", "train_daily_pnl.png"),
    ("training/direction_summary.png", "train_direction.png"),
    ("testing/pnl_hist.png", "test_pnl_hist.png"),
    ("testing/direction_summary.png", "test_direction.png"),
    ("model/losses.png", "train_losses.png"),
    ("testing/mae_mfe.png", "test_mae_mfe.png"),
    ("testing/hold_time_pnl.png", "test_hold_time.png"),
    ("testing/tod_heatmap.png", "test_tod_heatmap.png"),
    ("training/mae_mfe.png", "train_mae_mfe.png"),
    ("training/hold_time_pnl.png", "train_hold_time.png"),
    ("training/tod_heatmap.png", "train_tod_heatmap.png"),
)


def _copy_seed50(run_dir: Path, out: Path) -> None:
    for rel, dest in _COPY:
        src = run_dir / rel
        if src.exists():
            shutil.copy2(src, out / dest)


def _load_equity(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = df.columns[0]
    s = df[col].astype(float)
    s.index = pd.DatetimeIndex(s.index)
    return s


def _plot_risk_charts(run_dir: Path, out: Path, dpi: int) -> None:
    """Equity / DD / daily PnL with the $5,000 daily cap. No 7% trailing line."""
    log_path = run_dir / "training_log.json"
    fail = None
    if log_path.exists():
        payload = json.loads(log_path.read_text())
        fail = payload.get("test_fail_time")
    train_eq_path = run_dir / "training" / "equity.csv"
    if train_eq_path.exists():
        train_eq = _load_equity(train_eq_path)
        plot_drawdown(
            train_eq,
            daily_loss_limit=5000.0,
            out_path=out / "train_drawdown.png",
            dpi=dpi,
        )
        plot_daily_pnl(
            train_eq, daily_loss_limit=5000.0, out_path=out / "train_daily_pnl.png", dpi=dpi
        )
    eq_path = run_dir / "testing" / "equity.csv"
    if not eq_path.exists():
        return
    equity = _load_equity(eq_path)
    if fail:
        cut = pd.Timestamp(fail)
        eq_idx = pd.DatetimeIndex(equity.index)
        if eq_idx.tz is not None and cut.tzinfo is None:
            cut = cut.tz_localize(eq_idx.tz)
        equity = equity.loc[:cut]
    trades_path = run_dir / "testing" / "trades.csv"
    trades = pd.read_csv(trades_path, parse_dates=["time"]) if trades_path.exists() else None
    plot_equity_curve(
        equity,
        trades=trades,
        daily_loss_limit=5000.0,
        max_loss_limit=10000.0,
        profit_target=10000.0,
        initial_balance=100000.0,
        out_path=out / "test_equity.png",
        dpi=dpi,
    )
    plot_drawdown(
        equity,
        daily_loss_limit=5000.0,
        out_path=out / "test_drawdown.png",
        dpi=dpi,
    )
    plot_daily_pnl(equity, daily_loss_limit=5000.0, out_path=out / "test_daily_pnl.png", dpi=dpi)


def _pick_po3_day(features: pd.DataFrame, train_end: str) -> pd.Timestamp | None:
    idx = pd.DatetimeIndex(features.index)
    te = pd.Timestamp(train_end)
    if idx.tz is not None and te.tzinfo is None:
        te = te.tz_localize(idx.tz)
    te = te + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ny = np.asarray(ny_session_mask(idx))
    train = np.asarray(idx <= te)
    score = pd.Series(0.0, index=features.index)
    for col in ("price_in_ifvg_bull", "price_in_ifvg_bear", "entry_long", "entry_short"):
        if col in features.columns:
            score = score + features[col].fillna(0).astype(float).gt(0).astype(float)
    score = score.where(pd.Series(ny & train, index=features.index), 0.0)
    daily = score.groupby(idx.normalize()).sum()
    daily_ent = pd.Series(0.0, index=daily.index)
    for col in ("entry_long", "entry_short"):
        if col in features.columns:
            ent = features[col].fillna(0).astype(float).gt(0).astype(float)
            ent = ent.where(pd.Series(ny & train, index=features.index), 0.0)
            daily_ent = daily_ent.add(ent.groupby(idx.normalize()).sum(), fill_value=0.0)
    candidates = daily[(daily_ent >= 1) & (daily_ent <= 6) & (daily > 0)]
    if candidates.empty:
        candidates = daily[daily > 0]
    if candidates.empty:
        return None
    return pd.Timestamp(candidates.idxmin())


def _write_slides(out: Path) -> None:
    rows = [
        ("coverage_gaps.png", "M1 coverage and missing minutes vs 1440 weekday bars."),
        ("spread_regime.png", "Spread is bimodal: tight NY vs wide off-hours."),
        ("activity_24h.png", "Tick-count activity peaks in NY 16:30–23:00 (CFD vol unused)."),
        ("ny_return_tails.png", "NY M1 returns are heavy-tailed — risk-aware reward."),
        ("train_test_split.png", "Locked split: train ≤2025-12-31, OOS ≥2026-01-01."),
        ("feature_hists_curated.png", "Curated NY observation columns (clipped)."),
        ("event_rates.png", "IFVG/PO3 flags are rare; overlay is not every-bar."),
        ("ifvg_occupancy_monthly.png", "Monthly IFVG occupancy; watch 2026-01 shift."),
        ("ifvg_event_study.png", "Train purged event study vs random NY (pre-RL)."),
        ("pca_ifvg.png", "Technical PCA coloured by IFVG — honesty check, not proof."),
        ("po3_ny_annotated.png", "NY PO3 chain: Asian, IFVG, entry, SL, RR TP (not 3-scale)."),
        ("po3_fvg_sample.png", "Cleaned NY-window FVG/IFVG sample (not overnight soup)."),
        ("train_equity.png", "Seed 50 train: survived the year, +15.2%, 0 FTMO breaches."),
        ("train_drawdown.png", "Train peak-to-trough DD. Daily cap is −$5,000 (not 7%)."),
        ("train_daily_pnl.png", "Train daily PnL vs FTMO −$5,000 daily line."),
        ("train_direction.png", "Train: shorts earned, longs lost."),
        ("test_equity.png", "Test dies at $90k on 2026-04-10 (cropped at fail)."),
        ("test_drawdown.png", "Test drawdown vs −$5,000 daily cap (no 7% line)."),
        ("test_daily_pnl.png", "Test daily PnL vs −$5,000, cropped at fail date."),
        ("test_pnl_hist.png", "Test trade PnL: 75 wins / 140 losses."),
        ("train_losses.png", "PPO policy/value losses over 20M steps."),
        ("test_mae_mfe.png", "Appendix: test MAE/MFE."),
        ("test_hold_time.png", "Appendix: hold time vs PnL."),
        ("test_tod_heatmap.png", "Appendix: time-of-day PnL heatmap."),
    ]
    lines = [
        "# Defense figure catalog",
        "",
        "No slide-count cap. Trailing 7% DD is a **training** constraint only — not plotted.",
        "Daily loss on PnL and daily-drawdown charts is **$5,000** (not 7%). Do not use coverage_calendar /",
        "log_return_dist / feature_histograms / signal_counts / overnight po3_fvg_sample",
        "from the raw seed-50 `data/` folder; use the repaired files below.",
        "",
        "| File | Speaker line |",
        "| --- | --- |",
    ]
    for name, line in rows:
        if (out / name).exists():
            lines.append(f"| `{name}` | {line} |")
    (out / "slides.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=_SEED50)
    parser.add_argument("--out", type=Path, default=Path("outputs/defense_figures"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--count-ticks", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().handlers[0].flush if logging.getLogger().handlers else None
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path=Path("config/features_full_po3_mtf.yaml"))
    idea = Path("config/idea1_po3_ifvg.yaml")
    if idea.exists():
        cfg = cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.load(idea)))
    cfg.features.vwap_session = True
    cfg.features.include_session_ohlc = True
    train_end, test_start = get_split_config(cfg)
    data = run_pipeline(cfg, force=False)
    bars = attach_activity(data[str(cfg.data.primary)]["M1"])
    if args.count_ticks:
        tick_rel = cfg.data.tick_files[cfg.data.primary]
        tick_path = Path(cfg.data.raw_dir) / str(tick_rel)
        if tick_path.exists():
            from quant_rl.data.activity import load_tick_index

            tidx = load_tick_index(tick_path, tz=str(cfg.data.tz))
            bars["tick_count"] = tick_counts_from_index(cast(pd.DatetimeIndex, bars.index), tidx)
            bars = attach_activity(bars)
    cache_dir = Path(cfg.data.cache_dir)
    mask = make_train_mask(cast(pd.DatetimeIndex, bars.index), train_end)
    feat_path = feature_cache_path(cache_dir, str(cfg.data.primary), cfg, bars, train_mask=mask)
    if feat_path.exists():
        features = pd.read_parquet(feat_path)
        log.info("Loaded features %s %s", feat_path, features.shape)
    else:
        existing = sorted(cache_dir.glob(f"{cfg.data.primary}_features_*.parquet"))
        existing = [p for p in existing if p.suffix == ".parquet" and not p.name.endswith(".hash")]
        preferred = sorted(
            cache_dir.glob(f"{cfg.data.primary}_features_v12-trader-context*.parquet"),
            key=lambda p: p.stat().st_mtime,
        )
        if preferred:
            features = pd.read_parquet(preferred[-1])
            log.info("Loaded existing features %s %s", preferred[-1], features.shape)
        elif existing:
            existing.sort(key=lambda p: p.stat().st_mtime)
            features = pd.read_parquet(existing[-1])
            log.info("Loaded existing features %s %s", existing[-1], features.shape)
        else:
            secondary = data.get(str(cfg.data.secondary), {}).get("M1")
            features = build_features(
                bars,
                secondary=secondary,
                cfg=cfg,
                train_mask=mask,
                cache_path=feat_path,
                force=False,
            )
    dpi = int(args.dpi)
    plot_coverage_with_gaps(bars, out_path=out / "coverage_gaps.png", dpi=dpi)
    plot_spread_regime(
        bars,
        point_size=float(cfg.costs.point_size),
        out_path=out / "spread_regime.png",
        dpi=dpi,
    )
    plot_activity_24h(bars, out_path=out / "activity_24h.png", dpi=dpi)
    plot_ny_return_tails(bars, out_path=out / "ny_return_tails.png", dpi=dpi)
    plot_split_overlay(bars, train_end, test_start, out_path=out / "train_test_split.png", dpi=dpi)
    plot_curated_feature_hists(features, out_path=out / "feature_hists_curated.png", dpi=dpi)
    plot_event_rates(features, train_end, test_start, out_path=out / "event_rates.png", dpi=dpi)
    plot_ifvg_occupancy_monthly(
        features, test_start=test_start, out_path=out / "ifvg_occupancy_monthly.png", dpi=dpi
    )
    plot_ifvg_event_study(bars, features, train_end, out_path=out / "ifvg_event_study.png", dpi=dpi)
    plot_pca_ifvg(features, train_end, out_path=out / "pca_ifvg.png", dpi=dpi)
    day = _pick_po3_day(features, train_end)
    if day is not None:
        plot_annotated_ny_session(
            bars, features, day, out_path=out / "po3_ny_annotated.png", dpi=dpi
        )
        from quant_rl.eval.data_plots import _slice_ny_day

        plot_bars, lookback = _slice_ny_day(bars, day)
        sig = features.reindex(lookback.index)
        plot_fvg_signals(
            lookback,
            sig,
            window=(plot_bars.index[0], plot_bars.index[-1]) if len(plot_bars) else None,
            candle_tf="5min",
            max_points=400,
            out_path=out / "po3_fvg_sample.png",
            dpi=dpi,
            zone_kinds=("htf_fvg", "ltf_ifvg"),
            max_zones=8,
        )
    _copy_seed50(args.run_dir, out)
    _plot_risk_charts(args.run_dir, out, dpi)
    _write_slides(out)
    log.info("Defense figures written to %s", out)


if __name__ == "__main__":
    main()
