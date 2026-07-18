"""Tests for date-based train/test split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_rl.data.split import split_train_test, get_split_config


def _make_bars(start: str, end: str, freq: str = "1D", tz: str = "Etc/GMT-3") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq=freq, tz=tz)
    return pd.DataFrame({"close": np.random.rand(len(idx)), "session_id": 0}, index=idx)


def _make_feats(bars: pd.DataFrame, n_col: int = 5) -> pd.DataFrame:
    return pd.DataFrame(np.random.rand(len(bars), n_col).astype(np.float32), index=bars.index)


class TestSplitBoundaries:
    def setup_method(self):
        self.bars = pd.concat([
            _make_bars("2025-01-02", "2025-12-31"),
            _make_bars("2026-01-01", "2026-06-30"),
        ])
        self.feat = _make_feats(self.bars)

    def test_train_max_date(self):
        tr_b, _, _, _ = split_train_test(self.bars, self.feat, "2025-12-31", "2026-01-01")
        assert tr_b.index.max() <= pd.Timestamp("2025-12-31", tz="Etc/GMT-3") + pd.Timedelta(days=1)

    def test_test_min_date(self):
        _, te_b, _, _ = split_train_test(self.bars, self.feat, "2025-12-31", "2026-01-01")
        assert te_b.index.min() >= pd.Timestamp("2026-01-01", tz="Etc/GMT-3")

    def test_no_overlap(self):
        tr_b, te_b, _, _ = split_train_test(self.bars, self.feat, "2025-12-31", "2026-01-01")
        assert len(tr_b.index.intersection(te_b.index)) == 0

    def test_full_coverage(self):
        tr_b, te_b, _, _ = split_train_test(self.bars, self.feat, "2025-12-31", "2026-01-01")
        assert len(tr_b) + len(te_b) == len(self.bars)

    def test_feature_rows_match_bar_rows(self):
        tr_b, te_b, tr_f, te_f = split_train_test(self.bars, self.feat)
        assert len(tr_b) == len(tr_f)
        assert len(te_b) == len(te_f)
        assert tr_b.index.equals(tr_f.index)
        assert te_b.index.equals(te_f.index)


def test_empty_test_split():
    """If all data is before test_start, test split should be empty."""
    bars = _make_bars("2024-01-01", "2025-06-30")
    feat = _make_feats(bars)
    tr_b, te_b, _, _ = split_train_test(bars, feat, "2025-12-31", "2026-01-01")
    assert len(te_b) == 0
    assert len(tr_b) == len(bars)


def test_empty_train_split():
    """If all data is after train_end, train split should be empty."""
    bars = _make_bars("2026-02-01", "2026-06-30")
    feat = _make_feats(bars)
    tr_b, te_b, _, _ = split_train_test(bars, feat, "2025-12-31", "2026-01-01")
    assert len(tr_b) == 0
    assert len(te_b) == len(bars)


def test_get_split_config_fallback():
    """get_split_config should return defaults when cfg is None."""
    train_end, test_start = get_split_config(None)
    assert train_end  == "2025-12-31"
    assert test_start == "2026-01-01"


def test_m1_frequency():
    """Split should work correctly on M1 bars (not just daily)."""
    bars = pd.concat([
        _make_bars("2025-12-30", "2025-12-31", freq="1min"),
        _make_bars("2026-01-01", "2026-01-02", freq="1min"),
    ])
    feat = _make_feats(bars)
    tr_b, te_b, _, _ = split_train_test(bars, feat, "2025-12-31", "2026-01-01")
    assert not tr_b.empty
    assert not te_b.empty
    assert len(tr_b.index.intersection(te_b.index)) == 0
