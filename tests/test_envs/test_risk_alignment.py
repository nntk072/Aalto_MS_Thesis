"""Risk/sizing alignment, breakeven management, and agent direction control.

These cover the three changes in ``docs/thesis/agent_implementation_plan.md``:

1. The intraday adverse-excursion guard must never sit inside the trade's own
   structural stop, or a position sized to risk ``risk_per_trade_limit`` gets
   flattened early and the setup never plays out.
2. A trade that has paid one stop distance moves its stop to breakeven.
3. ``agent_direction_control`` gives the agent a signed side instead of the
   heuristic owning it (opt-in; the default layout is unchanged).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box

from quant_rl.backtest.broker import Position
from quant_rl.envs.strategies import PO3IFVGStrategy
from quant_rl.envs.trading_env import TradingEnv


def _bars(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01 16:30", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    close = np.clip(close, 99.0, 101.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "open": open_,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": np.zeros(n, dtype=int),
            "session": "ny",
        },
        index=idx,
    )


def _features(bars: pd.DataFrame, *, ctx_dir: float = 1.0) -> pd.DataFrame:
    n = len(bars)
    is_long = ctx_dir > 0
    return pd.DataFrame(
        {
            "asian_high": np.full(n, 102.0),
            "asian_low": np.full(n, 98.0),
            "london_high": np.full(n, 103.0),
            "london_low": np.full(n, 97.0),
            "sweep_high": np.zeros(n) if is_long else np.ones(n),
            "sweep_low": np.ones(n) if is_long else np.zeros(n),
            "po3_manipulation_low": np.full(n, 95.0),
            "po3_manipulation_high": np.full(n, 105.0),
            "po3_manipulation_end": np.zeros(n),
            "po3_distribution": np.zeros(n),
            "po3_distribution_direction": np.zeros(n),
            "ifvg_bull_low": np.full(n, 98.0),
            "ifvg_bull_high": np.full(n, 99.0),
            "price_in_ifvg_bull": np.ones(n) if is_long else np.zeros(n),
            "price_in_ifvg_bear": np.zeros(n) if is_long else np.ones(n),
            "ifvg_bear_low": np.zeros(n) if is_long else np.full(n, 99.0),
            "ifvg_bear_high": np.zeros(n) if is_long else np.full(n, 100.0),
            "ifvg_bull_active": np.ones(n) if is_long else np.zeros(n),
            "ifvg_bear_active": np.zeros(n) if is_long else np.ones(n),
            "last_swing_high": np.full(n, np.nan),
            "last_swing_low": np.full(n, np.nan),
            "htf_day_bias": np.full(n, ctx_dir),
            "context_trade_direction": np.full(n, ctx_dir),
            "manip_reverses_htf": np.zeros(n),
            "atr_5": np.full(n, 1.0),
        },
        index=bars.index,
    )


def _make_env(**kwargs: Any) -> TradingEnv:
    bars = _bars()
    strategy = kwargs.pop("strategy", None) or PO3IFVGStrategy(enforce_gate=False)
    kwargs.setdefault("obs_window", 10)
    kwargs.setdefault("risk_frac_range", (0.01, 0.01))
    kwargs.setdefault("rr_ratio_range", (2.0, 2.0))
    kwargs.setdefault("min_sl_points", 0.0)
    kwargs.setdefault("min_sl_atr_mult", 0.0)
    kwargs.setdefault("max_entries_per_session", 0)
    kwargs.setdefault("entry_cooldown_bars", 0)
    return TradingEnv(
        bars,
        _features(bars),
        strategy_actions=True,
        strategy=strategy,
        **kwargs,
    )


def _open_a_position(env: TradingEnv) -> None:
    """Step until a position exists, then return with it still open."""
    env.reset()
    for _ in range(40):
        env.step(np.array([0.9, 0.0, 0.5, 0.0], dtype=np.float32))
        if env.position is not None:
            return
    pytest.fail("expected a position to open")


def _risk_pts(pos: Position) -> float:
    """Stop distance in price points at entry, narrowing the optional stop."""
    return abs(pos.entry_price - _sl0(pos))


def _sl0(pos: Position) -> float:
    """Stop level as placed at entry, narrowing the optional stop."""
    assert pos.sl_initial_price is not None
    return float(pos.sl_initial_price)


def _sl(pos: Position) -> float:
    """Current stop level, narrowing the optional stop."""
    assert pos.sl_price is not None
    return float(pos.sl_price)


class TestEodGuardRespectsPlannedRisk:
    def test_planned_risk_is_recorded_at_entry(self) -> None:
        env = _make_env()
        _open_a_position(env)
        assert env.position is not None
        pos = env.position
        # The recorded figure is exactly what the stop would cost:
        # |entry - stop| * lots * contract_size.
        assert pos.planned_risk_usd is not None
        expected = _risk_pts(pos) * pos.size * env.contract_size
        assert pos.planned_risk_usd == pytest.approx(expected, rel=1e-9)
        assert pos.planned_risk_usd > 0.0

    def test_planned_risk_never_exceeds_the_sizing_budget(self) -> None:
        env = _make_env(max_loss_per_trade_usd=250.0)
        _open_a_position(env)
        assert env.position is not None
        assert float(env.position.planned_risk_usd or 0.0) <= 250.0 + 1e-6

    def test_cut_sits_outside_the_structural_stop(self) -> None:
        env = _make_env(eod_risk={"max_loss_usd": 500.0, "planned_risk_mult": 1.1})
        _open_a_position(env)
        assert env.position is not None
        cut = env._eod_adverse_cut_usd(env.position)
        planned = float(env.position.planned_risk_usd or 0.0)
        # The old $500 cap would have fired at half the planned risk.
        assert cut > planned
        assert cut >= 1.1 * planned

    def test_opt_out_restores_the_plain_configured_cap(self) -> None:
        env = _make_env(eod_risk={"max_loss_usd": 500.0, "respect_planned_risk": False})
        _open_a_position(env)
        assert env.position is not None
        assert env._eod_adverse_cut_usd(env.position) == pytest.approx(500.0)

    def test_large_configured_cap_still_wins(self) -> None:
        """The cut is max(configured, mult * planned): never below configured."""
        env = _make_env(eod_risk={"max_loss_usd": 5000.0, "planned_risk_mult": 1.1})
        _open_a_position(env)
        assert env.position is not None
        assert env._eod_adverse_cut_usd(env.position) == pytest.approx(5000.0)

    def test_stop_out_does_not_use_the_adverse_cut(self) -> None:
        """The structural stop remains the primary risk boundary."""
        env = _make_env(eod_risk={"max_loss_usd": 1100.0, "planned_risk_mult": 1.1})
        _open_a_position(env)
        assert env.position is not None
        pos = env.position
        assert pos.sl_price is not None
        reasons = [t.get("reason") for t in env.trade_log]
        assert "eod_max_loss" not in reasons


class TestBreakevenManagement:
    def test_disabled_by_default(self) -> None:
        env = _make_env()
        _open_a_position(env)
        assert env.position is not None
        assert env.position.sl_price == pytest.approx(_sl0(env.position))
        assert env.position.breakeven_done is False

    def test_stop_moves_to_breakeven_after_one_r(self) -> None:
        env = _make_env(eod_risk={"breakeven_trigger_r": 1.0, "breakeven_buffer_pts": 0.5})
        _open_a_position(env)
        assert env.position is not None
        pos = env.position
        risk_pts = _risk_pts(pos)
        env._apply_breakeven(pos, favorable=risk_pts + 0.01)
        assert pos.breakeven_done is True
        assert _sl(pos) > _sl0(pos)
        assert _sl(pos) == pytest.approx(pos.entry_price + 0.5)

    def test_below_the_trigger_nothing_changes(self) -> None:
        env = _make_env(eod_risk={"breakeven_trigger_r": 1.0, "breakeven_buffer_pts": 0.5})
        _open_a_position(env)
        assert env.position is not None
        pos = env.position
        risk_pts = _risk_pts(pos)
        env._apply_breakeven(pos, favorable=0.5 * risk_pts)
        assert pos.breakeven_done is False
        assert pos.sl_price == pytest.approx(_sl0(pos))

    def test_rewrite_only_ever_tightens(self) -> None:
        """Calling twice must not push the stop back out again."""
        env = _make_env(eod_risk={"breakeven_trigger_r": 1.0, "breakeven_buffer_pts": 0.5})
        _open_a_position(env)
        assert env.position is not None
        pos = env.position
        risk_pts = _risk_pts(pos)
        env._apply_breakeven(pos, favorable=risk_pts + 1.0)
        first = _sl(pos)
        env._apply_breakeven(pos, favorable=risk_pts + 5.0)
        assert _sl(pos) == pytest.approx(first)

    def test_short_side_moves_the_stop_down(self) -> None:
        env = _make_env(eod_risk={"breakeven_trigger_r": 1.0, "breakeven_buffer_pts": 0.5})
        _open_a_position(env)
        assert env.position is not None
        pos = env.position
        # Flip the fixture's bookkeeping to the short side and re-check.
        pos.direction = -1
        pos.sl_initial_price = pos.entry_price + 1.0
        pos.sl_price = pos.sl_initial_price
        pos.breakeven_done = False
        env._apply_breakeven(pos, favorable=1.5)
        assert pos.breakeven_done is True
        assert _sl(pos) == pytest.approx(pos.entry_price - 0.5)


class TestAgentDirectionControl:
    def test_default_action_space_stays_4d(self) -> None:
        env = _make_env()
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (4,)

    def test_opt_in_action_space_is_5d_and_signed(self) -> None:
        env = _make_env(agent_direction_control=True)
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (5,)
        assert float(env.action_space.low[0]) == pytest.approx(-1.0)
        assert float(env.action_space.high[0]) == pytest.approx(1.0)

    def test_agent_can_veto_the_heuristic_side(self) -> None:
        """Context is long, but strong negative conviction takes the short."""
        env = _make_env(agent_direction_control=True)
        row = env._feature_row_at(env.step_idx)
        # a[0] = -1 -> u[0] = 0 -> conviction -0.5 -> short.
        da, _, _, _ = env._decode_action(np.array([-1.0, 0.5, 0.5, 0.5, 0.5]), row)
        assert da == -1

    def test_agent_can_confirm_the_heuristic_side(self) -> None:
        env = _make_env(agent_direction_control=True)
        row = env._feature_row_at(env.step_idx)
        # a[0] = +1 -> u[0] = 1 -> conviction +0.5 -> long (context is long too).
        da, _, _, _ = env._decode_action(np.array([1.0, 0.5, 0.5, 0.5, 0.5]), row)
        assert da == 1

    def test_neutral_conviction_defers_to_the_heuristic(self) -> None:
        env = _make_env(agent_direction_control=True)
        row = env._feature_row_at(env.step_idx)
        # a[0] = 0 -> u[0] = 0.5 -> exactly neutral -> context_direction.
        da, _, _, _ = env._decode_action(np.array([0.0, 0.5, 0.5, 0.5, 0.5]), row)
        assert da == 1

    def test_positive_threshold_narrows_the_override_band(self) -> None:
        env = _make_env(agent_direction_control=True, direction_override_threshold=0.4)
        row = env._feature_row_at(env.step_idx)
        # conviction = -0.1, inside the 0.4 band -> defer to long context.
        da, _, _, _ = env._decode_action(np.array([-0.2, 0.5, 0.5, 0.5, 0.5]), row)
        assert da == 1

    def test_last_four_dims_keep_their_meaning(self) -> None:
        env = _make_env(agent_direction_control=True)
        row = env._feature_row_at(env.step_idx)
        # dims 1..4 are intensity, sl_anchor, risk, reward fraction.
        _, risk_frac, reward_frac, _ = env._decode_action(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), row)
        assert risk_frac == pytest.approx(0.01)
        assert reward_frac == pytest.approx(1.0)
        assert env._selected_sl_anchor == pytest.approx(1.0)

    def test_disabled_control_ignores_an_extra_dim(self) -> None:
        """A 5-D action on the 4-D layout must not shift the other dims."""
        env = _make_env()
        row = env._feature_row_at(env.step_idx)
        da_4, risk_4, reward_4, _ = env._decode_action(np.array([1.0, 1.0, 1.0, 1.0]), row)
        da_5, risk_5, reward_5, _ = env._decode_action(np.array([1.0, 1.0, 1.0, 1.0, 1.0]), row)
        assert da_4 == da_5
        assert risk_4 == pytest.approx(risk_5)
        assert reward_4 == pytest.approx(reward_5)
