"""Causal lifecycle and TradingEnv wiring tests for persistent entry state."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from quant_rl.envs.entry_state import (
    ENTRY_STATES,
    EntryEvidence,
    EntryStateMachine,
    entry_state_one_hot,
)
from quant_rl.envs.strategies import PO3IFVGStrategy
from quant_rl.envs.trading_env import TradingEnv


def _evidence(
    bar: int,
    *,
    detected: bool = False,
    invalidated: bool = False,
    arm: bool = False,
    retest: bool = False,
    direction: int = 0,
) -> EntryEvidence:
    return EntryEvidence(
        bar=bar,
        evidence_detected=detected,
        invalidated=invalidated,
        arm_condition=arm,
        retest_confirmed=retest,
        direction=direction,
        setup_types=("sweep",) if detected else (),
        setup_levels_snapshot=(),
    )


def test_fsm_transitions_keep_trigger_intent_separate_from_entry() -> None:
    machine = EntryStateMachine(candidate_max_age_bars=4)
    assert machine.observe(_evidence(100, detected=True, arm=True, direction=1)) == "CANDIDATE"
    assert machine.observe(_evidence(101, arm=True)) == "ARMED"

    candidate = machine.request_trigger()
    assert candidate is machine.candidate
    assert candidate is not None and candidate.direction == 1
    assert machine.state == "ARMED"

    machine.mark_entered()
    assert str(machine.state) == "IN_POSITION"
    machine.observe(_evidence(102, invalidated=True))
    assert str(machine.state) == "IN_POSITION"
    machine.mark_closed()
    assert str(machine.state) == "FLAT"
    assert machine.candidate is None


def test_fsm_requires_matching_retest_when_configured() -> None:
    machine = EntryStateMachine(candidate_max_age_bars=4, arm_requires_retest=True)
    machine.observe(_evidence(100, detected=True, direction=1))
    assert machine.observe(_evidence(101, arm=True, direction=1)) == "CANDIDATE"
    assert machine.observe(_evidence(102, arm=True, retest=True, direction=-1)) == "FLAT"

    machine.observe(_evidence(103, detected=True, direction=1))
    assert machine.observe(_evidence(104, arm=True, retest=True, direction=1)) == "ARMED"


def test_candidate_age_boundary_and_snapshot_are_causal() -> None:
    source = {"ifvg_bull_low": 98.5}
    evidence = EntryEvidence(
        bar=100,
        evidence_detected=True,
        invalidated=False,
        arm_condition=False,
        retest_confirmed=False,
        direction=1,
        setup_types=("sweep", "ifvg"),
        setup_levels_snapshot=tuple(source.items()),
    )
    machine = EntryStateMachine(candidate_max_age_bars=3)
    machine.observe(evidence)
    source["ifvg_bull_low"] = 1.0
    assert machine.candidate is not None
    assert machine.candidate.setup_levels_snapshot == (("ifvg_bull_low", 98.5),)
    assert machine.candidate.setup_types == ("sweep", "ifvg")

    assert machine.observe(_evidence(103)) == "CANDIDATE"  # age == max is valid
    assert machine.observe(_evidence(104)) == "FLAT"  # age == max + 1 expires


def test_later_invalidation_does_not_rewrite_prior_trigger_request() -> None:
    machine = EntryStateMachine(candidate_max_age_bars=5)
    machine.observe(_evidence(100, detected=True, direction=1))
    machine.observe(_evidence(101, arm=True, direction=1))

    requested = machine.request_trigger()
    assert requested is not None and requested.origin_bar == 100
    machine.observe(_evidence(102, detected=True, direction=-1))
    assert machine.state == "FLAT"
    assert requested.direction == 1


def _market_fixture(execution_direction: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = 40
    index = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = np.full(n, 100.0)
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 2000),
            "tickvol": np.full(n, 50),
            "spread": np.full(n, 0.6),
            "session_id": np.zeros(n, dtype=int),
        },
        index=index,
    )
    sweep_low = np.zeros(n)
    sweep_low[10] = 1.0
    sweep_high = np.zeros(n)
    if execution_direction == -1:
        sweep_high[12] = 1.0
    retest = np.zeros(n)
    retest[11] = 1.0
    features = pd.DataFrame(
        {
            "asian_high": np.full(n, 90.0),
            "asian_low": np.full(n, 110.0),
            "london_high": np.full(n, 90.0),
            "london_low": np.full(n, 110.0),
            "volume_spike": np.full(n, 2.0),
            "sweep_low": sweep_low,
            "sweep_high": sweep_high,
            "sweep_low_level": np.full(n, 95.0),
            "sweep_high_level": np.full(n, 110.0),
            "po3_manipulation_low": np.full(n, 95.0),
            "po3_manipulation_high": np.full(n, 105.0),
            "po3_manipulation_end": np.ones(n),
            "po3_manipulation_active": np.zeros(n),
            "po3_distribution": np.zeros(n),
            "po3_distribution_direction": np.zeros(n),
            "ifvg_bull_low": np.full(n, 98.0),
            "ifvg_bull_high": np.full(n, 99.0),
            "ifvg_bull_active": np.ones(n),
            "ifvg_retest_bull": retest,
            "ifvg_bear_low": np.full(n, 101.0),
            "ifvg_bear_high": np.full(n, 102.0),
            "ifvg_bear_active": np.zeros(n),
            "ifvg_retest_bear": np.zeros(n),
            "last_swing_high": np.full(n, 105.0),
            "last_swing_low": np.full(n, 95.0),
            "prev_day_high": np.full(n, 108.0),
            "prev_day_low": np.full(n, 90.0),
            "atr_5": np.ones(n),
        },
        index=index,
    )
    return bars, features


@pytest.mark.parametrize("execution_direction", [0, -1])
def test_env_uses_armed_candidate_after_per_bar_setup_disappears(
    execution_direction: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    bars, features = _market_fixture(execution_direction)
    env = TradingEnv(
        bars,
        features,
        obs_window=10,
        strategy=PO3IFVGStrategy(enforce_gate=True),
        strategy_actions=True,
        entry_state_machine=True,
        candidate_max_age_bars=5,
        entry_state_observation=True,
        open_manipulation_bars=0,
        entry_cooldown_bars=0,
        min_sl_atr_mult=0.0,
        min_sl_points=0.0,
        allow_agent_sl_mode=False,
        allow_ema_exit=False,
    )
    observation, _ = env.reset()
    assert observation["account"].shape == (10,)
    assert tuple(observation["account"][6:]) == entry_state_one_hot("FLAT")
    decoded_states: list[str] = []
    original_decode = env._decode_action

    def track_decoded_state(action: Any, feat_row: Any = None) -> tuple[int, float, float, str]:
        assert env.entry_state_machine is not None
        decoded_states.append(env.entry_state_machine.state)
        return original_decode(action, feat_row)

    monkeypatch.setattr(env, "_decode_action", track_decoded_state)

    action = np.array([0.0, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    observation, _, _, _, _ = env.step(action)  # bar 10: detect, return candidate
    assert env.entry_state_machine is not None
    assert env.entry_state_machine.state == "CANDIDATE"
    assert tuple(observation["account"][6:]) == entry_state_one_hot("CANDIDATE")
    candidate_observation = observation

    observation, _, _, _, _ = env.step(action)  # bar 11: setup false, retest arms
    candidate_obs_state = ENTRY_STATES[int(np.argmax(candidate_observation["account"][6:]))]
    assert decoded_states[-1] == candidate_obs_state
    assert str(env.entry_state_machine.state) == "ARMED"
    assert tuple(observation["account"][6:]) == entry_state_one_hot("ARMED")
    entry_setup = getattr(env.strategy, "entry_setup")
    assert callable(entry_setup)
    assert entry_setup(features.iloc[11], 1) is False
    armed_observation = observation

    observation, _, _, _, info = env.step(action)  # bar 12: request the stored long
    armed_obs_state = ENTRY_STATES[int(np.argmax(armed_observation["account"][6:]))]
    assert decoded_states[-1] == armed_obs_state
    assert decoded_states == ["FLAT", "CANDIDATE", "ARMED"]
    assert env.position is not None
    assert env.position.direction == 1
    assert env.position.entry_setup_types == ("ifvg", "sweep")
    assert env.position.entry_origin_bar == 10
    assert str(env.entry_state_machine.state) == "IN_POSITION"
    assert tuple(observation["account"][6:]) == entry_state_one_hot("IN_POSITION")
    assert info["entry_state"] == "IN_POSITION"
    open_trade = next(row for row in env.trade_log if row.get("type") == "open")
    assert open_trade["direction"] == 1
    assert open_trade["entry_origin_bar"] == 10

    assert env.position is not None
    env.position.sl_price = 101.0
    observation, _, _, _, close_info = env.step(action)
    assert env.position is None
    assert env.entry_state_machine.state == "FLAT"
    assert tuple(observation["account"][6:]) == entry_state_one_hot("FLAT")
    assert close_info["entry_state"] == "FLAT"


def test_a3_shape_and_action_space_are_unchanged() -> None:
    bars, features = _market_fixture()
    options: dict[str, Any] = {
        "obs_window": 10,
        "strategy": PO3IFVGStrategy(enforce_gate=True),
        "strategy_actions": True,
        "allow_agent_sl_mode": False,
    }
    legacy = TradingEnv(bars, features, **options)
    stateful = TradingEnv(bars, features, entry_state_machine=True, **options)
    hidden = TradingEnv(
        bars,
        features,
        entry_state_machine=True,
        entry_state_observation=False,
        **options,
    )
    raw = TradingEnv(
        bars,
        features,
        entry_state_machine=True,
        normalize_account=False,
        **options,
    )

    assert isinstance(legacy.observation_space, spaces.Dict)
    assert isinstance(hidden.observation_space, spaces.Dict)
    assert isinstance(stateful.observation_space, spaces.Dict)
    assert legacy.observation_space.spaces["account"].shape == (6,)
    assert hidden.observation_space.spaces["account"].shape == (6,)
    assert stateful.observation_space.spaces["account"].shape == (10,)
    raw_obs, _ = raw.reset()
    assert raw_obs["account"].shape == (10,)
    assert tuple(raw_obs["account"][6:]) == entry_state_one_hot("FLAT")
    assert legacy.action_space.shape == stateful.action_space.shape
    obs, _, _, _, info = legacy.step(np.zeros(5, dtype=np.float32))
    assert obs["account"].shape == (6,)
    assert "entry_state" not in info
    assert tuple(stateful.reset()[0]["account"][6:]) == entry_state_one_hot(ENTRY_STATES[0])


def test_multi_tp_and_sl_mode_remain_available_with_entry_fsm() -> None:
    bars, features = _market_fixture()
    env = TradingEnv(
        bars,
        features,
        obs_window=10,
        strategy=PO3IFVGStrategy(enforce_gate=True),
        strategy_actions=True,
        entry_state_machine=True,
        open_manipulation_bars=0,
        min_sl_atr_mult=0.0,
        min_sl_points=0.0,
        allow_agent_sl_mode=True,
        allow_multi_tp=True,
        allow_simplex=True,
        allow_ema_exit=False,
    )
    assert env.action_space.shape == (10,)
    action = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.8, 0.0, 0.0, -1.0])
    env.reset()
    env.step(action)
    env.step(action)
    env.step(action)

    assert env.position is not None
    assert env.position.sl_mode == "trailing"
    assert env.position.sl_mode_overridden
    assert env.position.n_tp >= 2
    active_targets = (
        env.position.tp1_price,
        env.position.tp2_price,
        env.position.tp3_price,
    )
    assert sum(target is not None for target in active_targets) >= 2
