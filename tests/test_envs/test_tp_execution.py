from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from simulation.environment.env import TradingEnv
from simulation.environment.market_view import BarView
from tests.test_envs._tp_mode_fixtures import (
    _make_deterministic_bars,
    _make_features,
    _make_tp_env,
)
from trading.execution.broker import Position


class TestTPExecution:
    def test_env_rejects_entry_filling_on_final_ny_bar(self) -> None:
        bars = _make_deterministic_bars().copy()
        bars["session"] = ["ny", "ny", "ny", *(["asian"] * (len(bars) - 3))]
        env = _make_tp_env(
            bars=bars,
            features=_make_features(bars),
            block_overnight=True,
        )
        env.reset(seed=42)
        env.step_idx = 1
        env._ny_pos = 1

        with (
            mock.patch.object(env, "_decode_action", return_value=(1, 0.01, 2.0, "fixed")),
            mock.patch.object(env, "_lock_unconfirmed_side", return_value=(1, 0.01)),
        ):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert env.position is None
        assert not any(row.get("type") == "open" for row in env.trade_log)
        assert env._entry_diag.get("rejected_overnight", 0) == 1

    def test_bar_close_entry_does_not_replay_earlier_entry_bar_extrema(self) -> None:
        env = _make_tp_env(block_overnight=False)
        env.reset(seed=42)
        env.step_idx = 20
        bar_time = env._bar_time(env.step_idx)
        bar = env._bar_at(env.step_idx)
        position = env.broker.open_position(
            env.account,
            env._bar_quote(bar),
            1.0,
            1,
        )
        assert position is not None
        position.entry_timestamp = bar_time + pd.Timedelta(minutes=1)
        position.sl_price = float(bar.low) + 0.01
        position.sl_initial_price = position.sl_price
        env.position = position

        with (
            mock.patch.object(env, "_progress_market", return_value=(*env._bar_quote(bar), None)),
            mock.patch.object(env, "_check_guardrails", return_value=(None, False, False, False)),
            mock.patch.object(env, "_apply_peak_trailing_dd", return_value=False),
            mock.patch.object(env, "_decode_action", return_value=(0, 0.0, 2.0, "fixed")),
            mock.patch.object(env, "_lock_unconfirmed_side", return_value=(0, 0.0)),
        ):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert env.position is position
        assert not any(row.get("type") == "stop_close" for row in env.trade_log)

    def test_env_rejects_entry_without_a_future_fill_bar(self) -> None:
        bars = _make_deterministic_bars()
        env = _make_tp_env(
            bars=bars,
            features=_make_features(bars),
            block_overnight=False,
        )
        env.step_idx = int(env._ny_indices[-1])
        env._ny_pos = len(env._ny_indices) - 1
        with (
            mock.patch.object(env, "_decode_action", return_value=(1, 0.01, 2.0, "fixed")),
            mock.patch.object(env, "_lock_unconfirmed_side", return_value=(1, 0.01)),
        ):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        assert env.position is None
        assert not any(row.get("type") == "open" for row in env.trade_log)
        assert env._entry_diag.get("rejected_terminal", 0) == 1

    def test_strict_eligibility_rejects_missing_ny_session(self) -> None:
        bars = _make_deterministic_bars().copy()
        bars["session"] = "asian"
        with pytest.raises(ValueError, match="NY|ny|New York"):
            _make_tp_env(
                bars=bars,
                features=_make_features(bars),
                strict_research_eligibility=True,
            )

    def test_last_ny_bar_is_session_boundary(self) -> None:
        env = _make_tp_env(block_overnight=True)
        env.reset(seed=42)
        env._ny_pos = len(env._ny_indices) - 1
        env.step_idx = int(env._ny_indices[-1])
        assert env._is_ny_session_boundary()

    def test_session_boundary_closes_at_current_quote(self) -> None:
        env = _make_tp_env(block_overnight=True)
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        env.position = pos
        bar = BarView(open=100.0, high=101.0, low=99.0, close=100.0, spread=60.0)
        later = BarView(open=105.0, high=106.0, low=104.0, close=105.0, spread=60.0)
        with (
            mock.patch.object(env, "_is_ny_session_boundary", return_value=True),
            mock.patch.object(env, "_bar_at", return_value=later),
        ):
            env._progress_market(
                bar,
                env._feature_row_at(env.step_idx),
                env._bar_time(env.step_idx),
                0,
                env.step_idx + 1,
            )
        assert env.position is None
        assert env.trade_log[-1]["type"] == "eod_close"
        assert env.trade_log[-1]["price"] == pytest.approx(100.0)

    def test_short_ema_exit_pays_ask_spread(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, -1)
        assert pos is not None
        pos.exit_mode = "ema_21"
        pos.stop_distance = 1.0
        pos.best_favorable = 2.0
        env.position = pos
        bar = BarView(open=100.0, high=100.5, low=98.0, close=100.0, spread=60.0)
        with mock.patch.object(env, "_ema21_at", return_value=99.0):
            env._maybe_ema_exit(bar, env.step_idx, env._bar_time(env.step_idx))
        assert env.position is None
        assert env.trade_log[-1]["reason"] == "ema_21_exit"
        assert env.trade_log[-1]["price"] == pytest.approx(100.6)

    def _enter_multi_tp(
        self,
        env: TradingEnv,
        *,
        size: float = 1.0,
        fractions: tuple[float, ...] = (0.3, 0.3, 0.4),
    ) -> Position:
        pos = env.broker.open_position(env.account, (100.0, 100.0), size, 1)
        assert pos is not None
        pos.sl_price = 99.0
        pos.sl_initial_price = 99.0
        pos.tp_initial_size = size
        pos.tp_lot_fractions = fractions
        pos.tp1_price = 101.0
        pos.tp2_price = 102.0
        pos.tp3_price = 103.0
        env.position = pos
        return pos

    @staticmethod
    def _tp_closes(env: TradingEnv) -> list[dict[str, Any]]:
        return [t for t in env.trade_log if t["type"] in ("tp_partial_close", "tp_close")]

    @staticmethod
    def _hit_bar(high: float) -> BarView:
        return BarView(open=high - 0.4, high=high, low=99.5, close=high - 0.2, spread=0.6)

    def test_partial_close_uses_initial_position_size(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)

        env._check_sl_tp(
            BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )
        assert pos.size == pytest.approx(0.7)
        assert pos.tp_hit_mask == 1
        assert pos.tp1_price is None
        closes = self._tp_closes(env)
        assert len(closes) == 1
        assert closes[0]["size"] == pytest.approx(0.3)  # 0.3 * tp_initial_size

        env._check_sl_tp(
            BarView(open=101.0, high=102.4, low=100.8, close=102.0, spread=0.6),
            fill_bid=102.0,
            fill_ask=102.6,
        )
        # Still 0.3 of the entry size; 0.3 of the remaining 0.7 would be 0.21.
        assert pos.size == pytest.approx(0.4)
        closes = self._tp_closes(env)
        assert len(closes) == 2
        assert closes[1]["size"] == pytest.approx(0.3)

    def test_cooldown_starts_after_position_completion_not_partial_fill(self) -> None:
        env = _make_tp_env(entry_cooldown_bars=5)
        pos = self._enter_multi_tp(env)
        env._cooldown_until_bar = 0
        env.step_idx = 20

        env._check_sl_tp(
            BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )
        assert pos.size == pytest.approx(0.7)
        assert env._cooldown_until_bar == 0

        env._check_sl_tp(
            BarView(open=100.0, high=100.5, low=98.5, close=99.0, spread=0.6),
            fill_bid=99.0,
            fill_ask=99.6,
        )
        assert env.position is None
        assert env._cooldown_until_bar == 25

    def test_partial_position_closes_at_session_end_as_one_lifecycle(self) -> None:
        env = _make_tp_env(block_overnight=True)
        pos = self._enter_multi_tp(env)
        pos.position_id = "session:partial"
        env.trade_log.append({"type": "open", "position_id": pos.position_id})
        env._check_sl_tp(
            BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )
        assert pos.size == pytest.approx(0.7)

        current = BarView(open=101.0, high=101.5, low=100.5, close=101.2, spread=0.6)
        next_bar = BarView(open=105.0, high=105.5, low=104.5, close=105.2, spread=0.6)
        with (
            mock.patch.object(env, "_is_ny_session_boundary", return_value=True),
            mock.patch.object(env, "_bar_at", return_value=next_bar),
        ):
            env._progress_market(
                current,
                env._feature_row_at(env.step_idx),
                env._bar_time(env.step_idx),
                0,
                env.step_idx + 1,
            )

        fills = [row for row in env.trade_log if row.get("fill_id")]
        assert [row["size"] for row in fills] == pytest.approx([0.3, 0.7])
        assert [row["position_complete"] for row in fills] == [False, True]
        assert fills[-1]["position_net_pnl"] == pytest.approx(sum(row["net_pnl"] for row in fills))
        assert env.position is None

    def test_partial_close_never_exceeds_initial_size(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)

        for high in (101.4, 102.4, 103.4):
            env._check_sl_tp(self._hit_bar(high), fill_bid=high, fill_ask=high + 0.6)

        closes = self._tp_closes(env)
        assert len(closes) == 3
        total = sum(t["size"] for t in closes)
        assert total == pytest.approx(1.0)
        assert total <= float(pos.tp_initial_size or pos.size) + 1e-9
        assert env.position is None

    def test_partial_close_rounding_and_final_residual(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        # Thirds round down twice; the final slot must absorb the 0.01 remainder.
        self._enter_multi_tp(env, fractions=(0.333, 0.333, 0.334))

        for high in (101.4, 102.4, 103.4):
            env._check_sl_tp(self._hit_bar(high), fill_bid=high, fill_ask=high + 0.6)

        closes = self._tp_closes(env)
        assert [t["size"] for t in closes] == pytest.approx([0.33, 0.33, 0.34])
        assert sum(t["size"] for t in closes) == pytest.approx(1.0)
        assert env.position is None

    def test_multi_tp_does_not_double_execute(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)

        bar = BarView(open=100.2, high=101.4, low=99.5, close=101.0, spread=0.6)
        env._check_sl_tp(bar, fill_bid=101.0, fill_ask=101.6)
        assert pos.tp_hit_mask == 1

        # A re-armed level beyond the bar must not pay a second time.
        pos.tp1_price = 101.0
        env._check_sl_tp(bar, fill_bid=101.0, fill_ask=101.6)

        assert len(self._tp_closes(env)) == 1
        assert pos.size == pytest.approx(0.7)
        assert pos.tp_hit_mask == 1

    def test_ambiguous_ohlc_uses_stop_before_any_target(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        self._enter_multi_tp(env)

        env._check_sl_tp(
            BarView(open=100.0, high=102.5, low=98.5, close=101.0, spread=0.6),
            fill_bid=101.0,
            fill_ask=101.6,
        )

        fills = [row for row in env.trade_log if row.get("fill_id")]
        assert [row["type"] for row in fills] == ["stop_close"]
        assert fills[0]["position_complete"] is True

    @pytest.mark.parametrize(
        ("direction", "stop", "open_price", "high", "low", "expected_sign"),
        [
            (1, 99.0, 97.0, 100.0, 96.0, "below"),
            (-1, 101.0, 103.0, 104.0, 100.0, "above"),
        ],
    )
    def test_gap_through_stop_fills_at_open_or_worse(
        self, direction, stop, open_price, high, low, expected_sign
    ) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, direction)
        assert pos is not None
        pos.position_id = f"gap:{direction}"
        pos.sl_price = stop
        env.position = pos

        env._check_sl_tp(
            BarView(open=open_price, high=high, low=low, close=100.0, spread=0.0),
            fill_bid=100.0,
            fill_ask=100.0,
        )

        fill = next(row for row in env.trade_log if row.get("fill_id"))
        if expected_sign == "below":
            assert fill["price"] <= open_price
        else:
            assert fill["price"] >= open_price

    def test_bar_crossing_multiple_targets_preserves_slot_quantities(self) -> None:
        env = _make_tp_env()
        env.reset(seed=42)
        self._enter_multi_tp(env)

        env._check_sl_tp(
            BarView(open=100.0, high=104.0, low=99.5, close=103.5, spread=0.6),
            fill_bid=103.5,
            fill_ask=104.1,
        )

        fills = [row for row in env.trade_log if row.get("fill_id")]
        assert [row["tp_slot"] for row in fills] == [1, 2, 3]
        assert [row["size"] for row in fills] == pytest.approx([0.3, 0.3, 0.4])
        assert [row["position_complete"] for row in fills] == [False, False, True]
        assert len({row["fill_id"] for row in fills}) == 3

    def test_single_target_tick_log_uses_delayed_fill_time(self) -> None:
        from market_data.loading.ticks import TickBook

        env = _make_tp_env(fill_delay_ms=5_000)
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        pos.position_id = "delayed:single"
        pos.entry_timestamp = env._bar_time(env.step_idx)
        pos.sl_price = 99.0
        env.position = pos
        start = pd.Timestamp(pos.entry_timestamp)
        fill_time = start + pd.Timedelta(seconds=20)
        env._tickbook = TickBook(
            np.asarray(
                [start.value, (start + pd.Timedelta(seconds=10)).value, fill_time.value],
                dtype=np.int64,
            ),
            np.asarray([100.0, 98.5, 98.0]),
            np.asarray([100.6, 99.1, 98.6]),
        )

        env._check_sl_tp(
            BarView(open=100.0, high=101.0, low=98.0, close=99.0, spread=0.6),
            fill_bid=99.0,
            fill_ask=99.6,
        )

        assert env.trade_log[-1]["time"] == fill_time

    def test_skipped_bar_replay_records_a_completed_stop_fill(self) -> None:
        env = _make_tp_env(block_overnight=False)
        env.reset(seed=42)
        pos = env.broker.open_position(env.account, (100.0, 100.0), 1.0, 1)
        assert pos is not None
        pos.position_id = "replay:stop"
        pos.entry_timestamp = env._bar_time(env.step_idx - 1)
        pos.sl_price = float(env._bar_at(env.step_idx).low) + 0.1
        pos.sl_initial_price = pos.sl_price
        pos.planned_risk_usd = 1.0
        env.position = pos
        env.trade_log.append({"type": "open", "position_id": pos.position_id})
        skipped_idx = env.step_idx

        env._replay_skipped_bars(skipped_idx, skipped_idx + 1)

        assert env.position is None
        fill = [row for row in env.trade_log if row.get("fill_id")]
        assert len(fill) == 1
        assert fill[0]["position_id"] == "replay:stop"
        assert fill[0]["position_complete"] is True

    def test_tick_tp1_is_partial_and_then_stop_closes_remainder(self) -> None:
        from market_data.loading.ticks import TickBook

        env = _make_tp_env()
        env.reset(seed=42)
        pos = self._enter_multi_tp(env)
        start = pd.Timestamp(env._bar_time(env.step_idx)).value
        seconds = np.array([1, 5, 10, 15], dtype=np.int64)
        env._tickbook = TickBook(
            start + seconds * 1_000_000_000,
            np.array([100.2, 101.2, 100.5, 98.8]),
            np.array([100.8, 101.8, 101.1, 99.4]),
        )

        env._check_sl_tp(
            BarView(open=100.2, high=101.2, low=98.8, close=99.1, spread=0.6),
            fill_bid=99.1,
            fill_ask=99.7,
        )
        closes = [r for r in env.trade_log if r.get("fill_id")]
        assert [r["type"] for r in closes] == ["tp_partial_close", "stop_close"]
        assert [r["size"] for r in closes] == pytest.approx([0.3, 0.7])
        assert closes[0]["tp_slot"] == 1
        assert closes[0]["price"] == pytest.approx(101.2)
        assert closes[1]["price"] == pytest.approx(98.8)
        assert pos.tp_hit_mask == 1
        assert env.position is None

    def test_tick_crosses_multiple_targets_in_one_minute(self) -> None:
        from market_data.loading.ticks import TickBook

        env = _make_tp_env()
        env.reset(seed=42)
        self._enter_multi_tp(env)
        start = pd.Timestamp(env._bar_time(env.step_idx)).value
        env._tickbook = TickBook(
            start + np.array([1, 7, 10], dtype=np.int64) * 1_000_000_000,
            np.array([100.1, 102.5, 103.3]),
            np.array([100.7, 103.1, 103.9]),
        )
        env._check_sl_tp(
            BarView(open=100.1, high=103.3, low=100.1, close=103.3, spread=0.6),
            fill_bid=103.3,
            fill_ask=103.9,
        )
        closes = self._tp_closes(env)
        assert [r["tp_slot"] for r in closes] == [1, 2, 3]
        assert [r["size"] for r in closes] == pytest.approx([0.3, 0.3, 0.4])
        assert sum(r["size"] for r in closes) == pytest.approx(1.0)
        assert env.position is None
