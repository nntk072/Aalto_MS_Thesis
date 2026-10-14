"""Causal structural target contracts, runnable without pytest or gymnasium."""

from __future__ import annotations

import unittest
from typing import Any

import pandas as pd

from quant_rl.backtest.entry_levels import OrderGeometry, place_strategy_order


class _Stops:
    def sl_candidates(self, *, direction: int, row: Any) -> list[tuple[str, float]]:
        return [
            (
                "last_swing_low" if direction == 1 else "last_swing_high",
                90.0 if direction == 1 else 110.0,
            )
        ]


def _place(row: pd.Series[Any], direction: int = 1, **options: Any) -> OrderGeometry:
    return place_strategy_order(
        _Stops(),
        direction=direction,
        entry_price=100.0,
        feat_row=row,
        min_dist=1.0,
        sl_fraction=0.0,
        tp_fraction=0.0,
        exit_mode="structural",
        buffer_pts=0.0,
        max_tp_distance=float("nan"),
        **options,
    )


class ReferenceTargetTest(unittest.TestCase):
    def test_legacy_default_preserves_exact_prices(self):
        order = _place(pd.Series({"last_swing_high": 120.12345}))
        self.assertEqual(order.tp_price, 120.12345)

    def test_long_buffer_and_reference_provenance(self):
        row = pd.Series(
            {
                "last_swing_high": 120.0,
                "last_swing_high_confirmed_at": "2026-01-02T10:00:00",
            },
            name=pd.Timestamp("2026-01-02T11:00:00"),
        )
        order = _place(row, target_front_run_pts=1.0, price_increment=0.5)
        self.assertEqual(order.tp_price, 119.0)
        self.assertEqual(order.tp_references[0].original_price, 120.0)
        self.assertEqual(order.tp_references[0].adjusted_price, 119.0)
        self.assertEqual(order.tp_references[0].timeframe, "M1")
        self.assertEqual(order.tp_references[0].confirmed_at, "2026-01-02T10:00:00")
        self.assertEqual(order.tp_references[0].reason, "selected")

    def test_reference_with_future_confirmation_is_excluded(self):
        row = pd.Series(
            {
                "last_swing_high": 120.0,
                "last_swing_high_confirmed_at": "2026-01-02T12:00:00",
                "prev_day_high": 140.0,
            },
            name=pd.Timestamp("2026-01-02T11:00:00"),
        )
        order = _place(row)
        self.assertEqual(order.tp_ref, "prev_day_high")

    def test_invalid_confirmation_metadata_is_not_assumed_causal(self):
        row = pd.Series(
            {
                "last_swing_high": 120.0,
                "last_swing_high_confirmed_at": "not-a-time",
                "prev_day_high": 140.0,
            },
            name=pd.Timestamp("2026-01-02T11:00:00"),
        )
        self.assertEqual(_place(row).tp_ref, "prev_day_high")

    def test_explicit_confirmation_requires_decision_timestamp(self):
        row = pd.Series(
            {
                "last_swing_high": 120.0,
                "last_swing_high_confirmed_at": "2026-01-02T10:00:00",
                "prev_day_high": 140.0,
            }
        )
        self.assertEqual(_place(row).tp_ref, "prev_day_high")

    def test_short_front_run_rounds_toward_entry(self):
        order = _place(
            pd.Series({"last_swing_low": 79.7}),
            direction=-1,
            target_front_run_pts=0.7,
            price_increment=0.5,
        )
        self.assertEqual(order.tp_price, 80.5)

    def test_rounding_rechecks_minimum_executable_distance(self):
        order = _place(
            pd.Series({"last_swing_high": 110.2}),
            target_front_run_pts=0.4,
            price_increment=1.0,
        )
        self.assertTrue(order.rejected)
        self.assertEqual(order.reject_kind, "tp")

    def test_rounding_deduplicates_target_prices(self):
        order = _place(
            pd.Series({"last_swing_high": 121.1, "prev_day_high": 121.2}),
            target_front_run_pts=0.5,
            price_increment=1.0,
        )
        self.assertEqual(order.n_tp, 1)
        self.assertEqual(order.tp_price, 120.0)

    def test_target_reference_records_rr_fallback_reason(self):
        order = _place(
            pd.Series({"last_swing_high": 120.0}),
            rr_bounds=(4.0, 5.0),
        )
        self.assertEqual(order.tp_price, 120.0)
        self.assertEqual(
            order.tp_references[0].fallback_reason,
            "no_structural_reference_within_rr_window",
        )


if __name__ == "__main__":
    unittest.main()
