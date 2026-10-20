from __future__ import annotations

import pytest

from configuration import load_config
from simulation.environment.env import TradingEnv
from trading.account.state import AccountState
from trading.execution.broker import Position
from trading.policy.config import TradingPolicyConfig, resolve_trading_policy
from trading.risk.guardrails import FTMOGuardrails
from training.setup.environments import _build_eval_common, _training_episodic


def test_repeated_policy_config_resolves_without_changing_default() -> None:
    legacy = resolve_trading_policy(load_config())
    assert legacy.mode == "legacy"
    repeated = resolve_trading_policy(load_config(["trading_policy.mode=repeated_payout"]))
    assert repeated.operating_balance == 100_000.0
    assert repeated.profit_target_balance == 110_000.0
    assert repeated.account_floor == 90_000.0
    assert repeated.max_agent_risk_frac == pytest.approx(0.009)


def test_fixed_session_equity_daily_floor_counts_unrealized_loss() -> None:
    acc = AccountState(policy_mode="repeated_payout")
    acc.balance = 108_000.0
    acc.update_equity(0.0)
    acc.reset_daily()
    assert acc.daily_floor(5_000.0) == 103_000.0
    acc.update_equity(-2_000.0)
    assert acc.canonical_daily_loss() == 2_000.0
    assert acc.remaining_daily_capacity(5_000.0) == 3_000.0
    acc.update_equity(3_000.0)
    assert acc.daily_floor(5_000.0) == 103_000.0
    assert acc.canonical_daily_loss() == 0.0


def test_repeated_daily_guardrail_uses_fixed_anchor() -> None:
    acc = AccountState(policy_mode="repeated_payout")
    acc.balance = 108_000.0
    acc.update_equity(0.0)
    acc.reset_daily()
    guard = FTMOGuardrails(daily_loss_limit=5_000.0)
    acc.update_equity(-4_999.0)
    assert not guard.check_daily(acc)
    acc.update_equity(-5_000.0)
    assert guard.check_daily(acc)


def test_payout_reset_reanchors_session_and_preserves_static_floor_capacity() -> None:
    acc = AccountState(policy_mode="repeated_payout")
    acc.balance = 110_250.0
    acc.update_equity(0.0)
    acc.reset_operating_balance(100_000.0)
    assert acc.balance == 100_000.0
    assert acc.equity == 100_000.0
    assert acc.session_start_equity == 100_000.0
    assert acc.remaining_account_capacity(90_000.0) == 10_000.0


def test_repeated_policy_rejects_one_percent_risk() -> None:
    with pytest.raises(ValueError, match="max agent risk"):
        load = load_config(
            [
                "trading_policy.mode=repeated_payout",
                "trading_policy.repeated_payout.max_agent_risk_frac=0.01",
            ]
        )
        resolve_trading_policy(load)


def _repeated_env_stub(*, reserve: float = 0.0, episodic: bool = False) -> TradingEnv:
    env = TradingEnv.__new__(TradingEnv)
    env.risk_mode = "dynamic"
    env.fixed_risk_usd = 0.0
    env.repeated_payout_policy = True
    env.trading_policy = TradingPolicyConfig(
        mode="repeated_payout",
        execution_reserve_usd=reserve,
    ).validate()
    env.guardrails = FTMOGuardrails(
        daily_loss_limit=5_000.0,
        max_loss_limit=10_000.0,
        risk_per_trade_limit=900.0,
        trailing_dd_limit=0.0,
    )
    env.account = AccountState(policy_mode="repeated_payout")
    env.episodic = episodic
    env.position = None
    env.cumulative_payouts = 0.0
    env.cycle_count = 1
    env.cycle_successes = 0
    env.cycle_failures = 0
    env.cycle_events = []
    env.payout_events = []
    env.trade_log = []
    env._last_requested_risk_usd = 0.0
    env._last_approved_risk_usd = 0.0
    env._last_execution_reserve_usd = 0.0
    env._payout_withdrawn_this_step = 0.0
    return env


def test_repeated_risk_budget_respects_execution_reserve() -> None:
    env = _repeated_env_stub(reserve=100.0)
    assert TradingEnv._risk_budget_usd(env, 0.009) == pytest.approx(800.0)
    assert env._last_requested_risk_usd == pytest.approx(900.0)
    assert env._last_execution_reserve_usd == pytest.approx(100.0)
    assert TradingEnv._risk_budget_usd(env, 0.0) == 0.0


def test_repeated_risk_budget_respects_remaining_account_and_daily_capacity() -> None:
    env = _repeated_env_stub(reserve=100.0)
    env.account.balance = 90_500.0
    env.account.equity = 90_500.0
    env.account.session_start_equity = 95_000.0
    # Daily and static-account capacities are both $500; reserve leaves $400.
    assert TradingEnv._risk_budget_usd(env, 0.009) == pytest.approx(400.0)


def test_repeated_eval_payout_records_overshoot_and_resets_operating_capital() -> None:
    env = _repeated_env_stub(episodic=False)
    env.account.balance = 110_250.0
    env.account.equity = 110_250.0
    assert TradingEnv._handle_repeated_payout(env, "2026-01-02T12:00:00") is False
    assert env.cumulative_payouts == pytest.approx(10_250.0)
    assert env._payout_withdrawn_this_step == pytest.approx(10_250.0)
    assert env.account.balance == 100_000.0
    assert env.account.session_start_equity == 100_000.0
    assert env.cycle_count == 2
    assert env.cycle_successes == 1
    assert env.payout_events[0]["payout"] == pytest.approx(10_250.0)


def test_repeated_training_payout_is_terminal_without_rewinding_account() -> None:
    env = _repeated_env_stub(episodic=True)
    env.account.balance = 110_100.0
    env.account.equity = 110_100.0
    assert TradingEnv._handle_repeated_payout(env, "2026-01-02T12:00:00") is True
    assert env.account.balance == 110_100.0
    assert env.cycle_successes == 1


def test_repeated_training_and_eval_use_v3_schema_and_single_cycle_training() -> None:
    cfg = load_config(
        [
            "trading_policy.mode=repeated_payout",
            "costs.use_tick_execution=false",
            "execution.randomized_fill_latency_ms=0.0",
        ]
    )
    assert _training_episodic(cfg) is True
    common, *_ = _build_eval_common(cfg)
    assert common["observation_schema_version"] == "account-state-v3-repeated-payout"
    assert common["risk_frac_range"] == pytest.approx((0.0, 0.009))
    assert common["trading_policy"]["mode"] == "repeated_payout"
    assert common["guardrail_kwargs"]["trailing_dd_limit"] == 0.0


def test_repeated_policy_direct_and_bundle_worker_parity(tmp_path) -> None:
    import numpy as np

    from simulation.environment.bundles import build_env_bundle
    from simulation.environment.spec import EnvSpec
    from simulation.environment.worker import make_bundle_env
    from tests.test_envs.test_trading_env_golden_traces import (
        _make_deterministic_bars,
        _make_features,
    )
    from trading.strategies import BaselineStrategy
    from training.setup.environments import make_env

    bars = _make_deterministic_bars(n=96)
    features = _make_features(bars)
    features["atr_5"] = np.linspace(1.0, 3.0, len(bars))
    cfg = load_config(
        [
            "trading_policy.mode=repeated_payout",
            "env.strategy_actions=true",
            "strategy.name=baseline",
            "costs.use_tick_execution=false",
            "execution.randomized_fill_latency_ms=0.0",
        ]
    )
    bundle_path = build_env_bundle(
        tmp_path,
        bars,
        features,
        cfg,
        raw_columns=list(BaselineStrategy().raw_columns),
        feature_config_hash="repeated-policy-worker-parity-v1",
        tickbook=None,
        pre_ny_by_date=None,
    )
    spec = EnvSpec.from_cfg(
        cfg,
        algo="ppo",
        reward="dsr",
        arch="tcn",
        bundle_dir=str(bundle_path),
    )
    direct = make_env(bars, features, cfg, algo="ppo", reward="dsr", arch="tcn")
    bundled = make_bundle_env(spec)
    assert direct.trading_policy == bundled.trading_policy
    assert direct.risk_frac_range == bundled.risk_frac_range == (0.0, 0.009)
    assert direct.guardrails.daily_loss_limit == bundled.guardrails.daily_loss_limit == 5_000.0
    assert direct.guardrails.max_loss_limit == bundled.guardrails.max_loss_limit == 10_000.0
    assert direct.guardrails.trailing_dd_limit == bundled.guardrails.trailing_dd_limit == 0.0
    assert (
        direct.observation_schema.version
        == bundled.observation_schema.version
        == "account-state-v3-repeated-payout"
    )
    assert direct.action_space == bundled.action_space


def test_repeated_policy_rejects_discrete_risk_layout() -> None:
    from tests.test_envs.test_trading_env_golden_traces import (
        _make_deterministic_bars,
        _make_features,
    )
    from training.setup.environments import make_env

    bars = _make_deterministic_bars(n=96)
    features = _make_features(bars)
    cfg = load_config(
        [
            "trading_policy.mode=repeated_payout",
            "env.strategy_actions=false",
            "costs.use_tick_execution=false",
            "execution.randomized_fill_latency_ms=0.0",
        ]
    )
    with pytest.raises(ValueError, match="continuous risk action coordinate"):
        make_env(bars, features, cfg, algo="ppo", reward="dsr", arch="tcn")


def test_static_account_floor_breaches_at_exact_ninety_thousand() -> None:
    acc = AccountState(policy_mode="repeated_payout")
    guard = FTMOGuardrails(max_loss_limit=10_000.0)
    acc.balance = 90_000.01
    acc.update_equity(0.0)
    assert not guard.check_max_drawdown(acc)
    acc.balance = 90_000.0
    acc.update_equity(0.0)
    assert guard.check_max_drawdown(acc)
    acc.balance = 89_500.0
    acc.update_equity(0.0)
    assert guard.check_max_drawdown(acc)


def test_unrealized_target_touch_does_not_trigger_payout() -> None:
    env = _repeated_env_stub(episodic=False)
    env.account.balance = 109_000.0
    env.account.equity = 110_500.0
    env.account.open_pnl = 1_500.0
    env.position = Position(direction=1, size=1.0, entry_price=1.0, margin_used=1.0)
    assert TradingEnv._handle_repeated_payout(env, "2026-01-02T12:00:00") is False
    assert env.cumulative_payouts == 0.0
    assert env.cycle_successes == 0


def test_multiple_payouts_accumulate_and_advance_cycles() -> None:
    env = _repeated_env_stub(episodic=False)
    env.account.balance = 110_250.0
    env.account.equity = 110_250.0
    assert TradingEnv._handle_repeated_payout(env, "2026-01-02T12:00:00") is False
    env.account.balance = 110_100.0
    env.account.equity = 110_100.0
    assert TradingEnv._handle_repeated_payout(env, "2026-01-03T12:00:00") is False
    assert env.cumulative_payouts == pytest.approx(20_350.0)
    assert env.cycle_successes == 2
    assert env.cycle_count == 3
    assert [event["cycle"] for event in env.payout_events] == [1, 2]


def test_failure_after_payout_preserves_cumulative_payout() -> None:
    env = _repeated_env_stub(episodic=False)
    env.breach_log = []
    env.breach_events = []
    env.sessions_with_trades = set()
    env.account.balance = 110_250.0
    env.account.equity = 110_250.0
    TradingEnv._handle_repeated_payout(env, "2026-01-02T12:00:00")
    env.account.balance = 89_999.0
    env.account.equity = 89_999.0
    reason, blocked, done, truncated = TradingEnv._check_guardrails(
        env,
        "2026-01-03T12:00:00",
        20000,
        89_999.0,
        89_999.0,
    )
    assert reason == "max_drawdown"
    assert blocked is False
    assert done is True
    assert truncated is False
    assert env.cycle_failures == 1
    assert env.cumulative_payouts == pytest.approx(10_250.0)
    assert env.cycle_events[-1]["cumulative_payouts"] == pytest.approx(10_250.0)


def test_execution_reserve_can_consume_all_remaining_capacity() -> None:
    env = _repeated_env_stub(reserve=500.0)
    env.account.balance = 90_500.0
    env.account.equity = 90_500.0
    env.account.session_start_equity = 95_000.0
    assert TradingEnv._risk_budget_usd(env, 0.009) == 0.0
    assert env._last_requested_risk_usd > 0.0
    assert env._last_approved_risk_usd == 0.0
