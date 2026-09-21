"""Smoke tests for defense EDA charts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.eval.defense_plots import (
    plot_activity_24h,
    plot_coverage_with_gaps,
    plot_event_rates,
    plot_ifvg_event_study,
    plot_ifvg_occupancy_monthly,
    plot_ny_return_tails,
    plot_pca_ifvg,
    plot_spread_regime,
)
from quant_rl.eval.po3_plots import plot_annotated_ny_session


def _bars(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    rng = np.random.default_rng(0)
    close = 20000.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tickvol": rng.integers(10, 80, n).astype(float),
            "volume": np.zeros(n),
            "spread": np.where(np.arange(n) < 50, 150.0, 80.0),
        },
        index=idx,
    )


def test_defense_core_pngs(tmp_path) -> None:
    bars = _bars()
    plot_coverage_with_gaps(bars, out_path=tmp_path / "cov.png", dpi=72)
    plot_spread_regime(bars, out_path=tmp_path / "spr.png", dpi=72)
    plot_activity_24h(bars, out_path=tmp_path / "act.png", dpi=72)
    plot_ny_return_tails(bars, out_path=tmp_path / "ret.png", dpi=72)
    for name in ("cov.png", "spr.png", "act.png", "ret.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_pre_rl_pngs(tmp_path) -> None:
    bars = _bars()
    rng = np.random.default_rng(1)
    feat = pd.DataFrame(
        {
            "rsi": rng.uniform(20, 80, len(bars)),
            "atr": rng.uniform(1, 5, len(bars)),
            "price_in_ifvg_bull": rng.integers(0, 2, len(bars)),
            "po3_distribution": rng.integers(0, 2, len(bars)),
            "entry_long": rng.integers(0, 2, len(bars)),
        },
        index=bars.index,
    )
    plot_event_rates(feat, "2025-01-02", "2025-01-03", out_path=tmp_path / "er.png", dpi=72)
    plot_ifvg_occupancy_monthly(feat, out_path=tmp_path / "occ.png", dpi=72)
    plot_ifvg_event_study(bars, feat, "2025-12-31", horizon=5, out_path=tmp_path / "es.png", dpi=72)
    plot_pca_ifvg(feat, "2025-12-31", max_points=200, out_path=tmp_path / "pca.png", dpi=72)
    for name in ("er.png", "occ.png", "es.png", "pca.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_annotated_ny_session(tmp_path) -> None:
    bars = _bars()
    feat = pd.DataFrame(
        {
            "entry_long": 0.0,
            "asian_high": 20010.0,
            "asian_low": 19990.0,
            "po3_manipulation_low": 19980.0,
            "price_in_ifvg_bull": 0.0,
        },
        index=bars.index,
    )
    feat.loc[feat.index[10], "entry_long"] = 1.0
    fig = plot_annotated_ny_session(
        bars,
        feat,
        pd.Timestamp("2025-01-02", tz="Etc/GMT-3"),
        out_path=tmp_path / "po3.png",
        dpi=72,
        candle_tf="1min",
    )
    assert (tmp_path / "po3.png").stat().st_size > 0
    assert len(fig.axes) == 1
