"""Verify Idea 1 (PO3 + IFVG) strategy features (Agent.md §32, §33).

Loads M1 bars, builds the feature matrix with the strategy-state block
enabled, and prints a verification report: new columns present, event
counts, and NaN/inf checks. Zero-event counts are flagged as blockers.

Usage:
    .venv/bin/python scripts/verify_strategy_features.py
    .venv/bin/python scripts/verify_strategy_features.py --bars-csv data/us100_2025.csv

This script validates code only; it must not start training.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_rl.config import load_config
from quant_rl.data.clean import clean
from quant_rl.features.build import build_features

STRATEGY_COLUMNS = [
    "asian_high",
    "asian_low",
    "asian_range",
    "price_to_asian_high_atr",
    "price_to_asian_low_atr",
    "sweep_high",
    "sweep_low",
    "sweep_high_level",
    "sweep_low_level",
    "sweep_high_reclaimed",
    "sweep_low_reclaimed",
    "bos_up",
    "bos_down",
    "po3_manipulation_active",
    "po3_manipulation_high",
    "po3_manipulation_low",
    "po3_manipulation_end",
    "po3_distribution",
    "po3_distribution_direction",
    "ifvg_bull_active",
    "ifvg_bull_low",
    "ifvg_bull_high",
    "price_in_ifvg_bull",
    "ifvg_bear_active",
    "ifvg_bear_low",
    "ifvg_bear_high",
    "price_in_ifvg_bear",
    "manipulation_low_distance_atr",
    "manipulation_high_distance_atr",
]


def _synthetic_bars(n: int = 2000) -> pd.DataFrame:
    """Fallback bars when no real data file is available."""
    idx = pd.date_range("2024-01-02 01:05", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    high = close + np.abs(rng.normal(0, 0.05, n))
    low = close - np.abs(rng.normal(0, 0.05, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tickvol": rng.integers(100, 1000, n),
            "volume": rng.integers(1000, 5000, n),
            "session_id": (np.arange(n) // 200).astype(int),
        },
        index=idx,
    )


def _load_bars(bars_csv: str | None) -> pd.DataFrame:
    if bars_csv is None or not Path(bars_csv).exists():
        if bars_csv is not None:
            print(f"[warn] {bars_csv} not found — using synthetic bars", file=sys.stderr)
        return _synthetic_bars()
    df = pd.read_csv(bars_csv)
    return clean(df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-csv", default=None, help="US100 M1 CSV (data/us100_2025.csv)")
    parser.add_argument("--strategy", default="po3_ifvg", choices=["po3_ifvg", "distribution"])
    args = parser.parse_args()

    cfg = load_config(
        config_path="config/idea1_po3_ifvg.yaml"
        if args.strategy == "po3_ifvg"
        else "config/idea2_distribution.yaml"
    )

    bars = _load_bars(args.bars_csv)
    features = build_features(bars, cfg=cfg)

    print("FEATURE VERIFICATION")
    print("-" * 40)
    print(f"rows: {len(features)}")
    print(f"columns: {len(features.columns)}")
    print()

    all_ok = True
    for col in STRATEGY_COLUMNS:
        status = "OK" if col in features.columns else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"{col}: {status}")
    print()

    event_cols: dict[str, Callable[[pd.DataFrame], int]] = {
        "asian_high": lambda f: int(f["asian_high"].notna().sum()),
        "asian_low": lambda f: int(f["asian_low"].notna().sum()),
        "sweep_high": lambda f: int(f["sweep_high"].sum()),
        "sweep_low": lambda f: int(f["sweep_low"].sum()),
        "po3_manipulation_end": lambda f: int(f["po3_manipulation_end"].sum()),
        "po3_distribution": lambda f: int(f["po3_distribution"].sum()),
        "ifvg_bull_active": lambda f: int(f["ifvg_bull_active"].sum()),
        "ifvg_bear_active": lambda f: int(f["ifvg_bear_active"].sum()),
    }
    print("EVENT COUNTS")
    print("-" * 40)
    for col, fn in event_cols.items():
        if col in features.columns:
            count = fn(features)
            flag = "  <-- ZERO (explain before training)" if count == 0 else ""
            print(f"{col}: {count}{flag}")
        else:
            print(f"{col}: (column missing)")
    print()

    nan_cols = features.columns[features.isna().any()].tolist()
    inf_cols = [c for c in features.columns if np.isinf(features[c].to_numpy(dtype=float)).any()]
    print(f"NaN columns: {len(nan_cols)}")
    print(f"Inf columns: {len(inf_cols)}")

    model_features = features.dropna(how="all")
    nan_model = int(model_features.isna().sum().sum())
    print(f"NaN model features (post-dropna): {nan_model}")

    out_dir = Path("outputs/feature_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "strategy_features.parquet")
    print(f"\nsaved: {out_dir / 'strategy_features.parquet'}")

    # Warmup NaNs (rolling zscore / ATR windows) are expected; only Inf and
    # missing strategy columns are hard failures.
    return 0 if all_ok and len(inf_cols) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
