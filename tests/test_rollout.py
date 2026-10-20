"""Tests for evaluation.rollout.model.evaluate_model (RL model walk-forward eval).

Regression coverage for the bug where RL runs reported ``equity == 100k``
for every case with zero trades executed: the evaluation step used a
hardcoded hold-forever stub instead of querying the trained model. These
tests exercise the real fix — rolling a model through
``TradingEnv(episodic=False)`` — with a scripted stand-in model so no actual
PPO training is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from decisioning.actions import build_discrete_trading_action_schema
from decisioning.adapters.sb3 import SB3DecisionProvider
from decisioning.observations import OBSERVATION_SCHEMA_V1
from gymnasium import spaces

from evaluation.rollout.model import evaluate_model, evaluate_provider


class _ScriptedModel:
    """Minimal stand-in for a trained SB3 model's ``.predict()`` API.

    Ignores the observation and returns a scripted action per call index,
    mirroring ``model.predict(obs, deterministic=True) -> (action, state)``.
    """

    def __init__(self, script: dict[int, int], default: int = 0):
        self.script = script
        self.default = default
        self.t = 0
        self.action_space: spaces.Space[Any] = spaces.Discrete(20)
        self._action_schema: dict[str, object] = build_discrete_trading_action_schema().to_dict()

    def predict(self, obs, deterministic: bool = True):
        action = self.script.get(self.t, self.default)
        self.t += 1
        return np.array([action]), None


def _make_bars(n: int, session_split: int | None = None) -> pd.DataFrame:
    """Flat-price synthetic M1 bars (deterministic transaction costs)."""
    idx = pd.date_range("2025-01-06 16:30", periods=n, freq="1min", tz="Etc/GMT-3")
    close = np.full(n, 20000.0)
    session_id = np.zeros(n, dtype=int)
    if session_split is not None:
        session_id[session_split:] = 1
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "session_id": session_id,
        },
        index=idx,
    )


def _make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Trivial feature matrix aligned 1:1 with *bars*.

    Provides swing levels so entries compute structure SL/TP (the env no
    longer opens naked lots=1.0 positions when swing levels are missing).
    """
    n = len(bars)
    return pd.DataFrame(
        {
            "ret_1": np.zeros(n),
            "last_swing_low": np.full(n, 19990.0),
            "last_swing_high": np.full(n, 20010.0),
        },
        index=bars.index,
    )


def test_evaluate_model_produces_trades_and_valid_equity():
    """A model that enters/holds/exits repeatedly must produce real trades,
    not the flat 100k-equity/zero-trade result the hardcoded hold-stub gave.
    """
    obs_window = 10
    n = 120
    bars = _make_bars(n)
    features = _make_features(bars)

    # Enter long every 20 calls, exit 5 calls later, hold otherwise.
    script = {t: (1 if t % 20 == 0 else 19 if t % 20 == 5 else 0) for t in range(n - obs_window)}
    model = _ScriptedModel(script, default=0)

    result = evaluate_model(
        model,
        bars=bars,
        features=features,
        obs_window=obs_window,
        initial_balance=100_000.0,
    )

    trades = result["trades"]
    assert not trades.empty, "evaluate_model must produce real trades from a trading model"
    assert (trades["type"] == "open").sum() > 0

    equity = result["equity"]
    assert isinstance(equity, pd.Series)
    assert len(equity) == n - obs_window + 1
    assert isinstance(equity.index, pd.DatetimeIndex)
    assert result["completed_data_split"] is True
    assert result["episode_end_reason"] == "data_end"

    close_rows = trades[trades["type"] == "close"]
    assert not close_rows.empty
    assert close_rows["price"].notna().all()
    assert close_rows["bar"].notna().all()
    # Equity must not stay flat at the initial balance the whole rollout.
    assert not np.isclose(equity.iloc[-1], 100_000.0)


def test_evaluate_provider_uses_the_same_trading_rollout():
    obs_window = 10
    n = 120
    bars = _make_bars(n)
    features = _make_features(bars)
    script = {t: (1 if t % 20 == 0 else 19 if t % 20 == 5 else 0) for t in range(n - obs_window)}
    provider = SB3DecisionProvider(
        _ScriptedModel(script),
        input_schema_version=OBSERVATION_SCHEMA_V1,
    )

    result = evaluate_provider(
        provider,
        bars=bars,
        features=features,
        obs_window=obs_window,
        initial_balance=100_000.0,
    )

    assert not result["trades"].empty
    assert result["completed_data_split"] is True
    assert result["equity"].index[0] == bars.index[obs_window]
    assert len(result["equity"]) == len(bars) - obs_window + 1


class _ExposureEnv:
    def __init__(self) -> None:
        self.exposure = iter((True, True, False))
        self.constraint_steps = iter(
            (
                {"daily_cost": 0.1},
                {"trailing_cost": 0.2},
                {"max_loss_cost": 0.1},
            )
        )
        self.equity_curve = [100_000.0]
        self.equity_times = [pd.Timestamp("2025-01-06T16:40:00+03:00")]
        self.trade_log: list[dict[str, Any]] = []
        self.breach_events: list[dict[str, Any]] = []
        self.breach_log: list[dict[str, Any]] = []
        self.all_sessions = {"s0"}
        self.sessions_with_trades: set[str] = set()
        self.breached_sessions: set[str] = set()
        self.account = object()
        self._entry_diag: dict[str, int] = {}
        self.reset_seed: int | None = None

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        self.reset_seed = seed
        return {}, {}

    def step(
        self, action: object
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        del action
        is_open = next(self.exposure)
        self.equity_curve.append(100_000.0)
        self.equity_times.append(self.equity_times[-1] + pd.Timedelta(minutes=1))
        completed = len(self.equity_curve) == 4
        info: dict[str, object] = {
            "position": is_open,
            "constraints": next(self.constraint_steps),
        }
        if completed:
            info["episode_equity"] = {
                "completed_data_split": True,
                "episode_end_reason": "data_end",
            }
        return {}, 0.0, completed, False, info


def test_evaluate_model_reports_bar_close_exposure_fraction() -> None:
    result = evaluate_model(
        _ScriptedModel({}),
        bars=pd.DataFrame(),
        features=pd.DataFrame(),
        env=_ExposureEnv(),  # type: ignore[arg-type]
    )

    assert result["n_steps"] == 3
    assert result["completed_data_split"] is True
    assert result["survived_full_year"] is True
    assert result["exposure_bars"] == 2
    assert result["exposure_fraction"] == 2 / 3
    assert result["exposure_measure"] == "fraction_of_evaluation_steps_with_position_at_bar_close"


def test_evaluate_model_seeds_environment_execution_rng() -> None:
    env = _ExposureEnv()

    evaluate_model(
        _ScriptedModel({}),
        bars=pd.DataFrame(),
        features=pd.DataFrame(),
        env=env,  # type: ignore[arg-type]
        randomized_fill_latency_ms=30.0,
        execution_seed=17,
    )

    assert env.reset_seed == 17


def test_randomized_tick_evaluation_requires_execution_seed() -> None:
    with pytest.raises(ValueError, match="execution_seed is required"):
        evaluate_model(
            _ScriptedModel({}),
            bars=pd.DataFrame(),
            features=pd.DataFrame(),
            env=_ExposureEnv(),  # type: ignore[arg-type]
            randomized_fill_latency_ms=30.0,
        )


def test_q5_evaluation_reports_discounted_constraint_costs() -> None:
    model = _ScriptedModel({})
    setattr(model, "_constrained_rl_state", {"version": 2, "gamma": 0.9})
    result = evaluate_model(
        model,
        bars=pd.DataFrame(),
        features=pd.DataFrame(),
        env=_ExposureEnv(),  # type: ignore[arg-type]
    )

    summary = result["constraint_cost_summary"]
    assert summary["schema"] == "q5_discounted_constraint_cost_v1"
    assert summary["n_steps"] == 3
    assert summary["undiscounted_total"] == pytest.approx(0.4)
    assert summary["discounted_total"] == pytest.approx(0.361)
    assert summary["components"]["trailing_cost"]["discounted_total"] == pytest.approx(0.18)
    assert result["constraint_costs"]["discount_factor"].tolist() == pytest.approx([1.0, 0.9, 0.81])


def test_evaluate_model_does_not_mark_step_capped_rollout_as_full_survival():
    bars = _make_bars(120)
    features = _make_features(bars)

    result = evaluate_model(
        _ScriptedModel({}, default=0),
        bars=bars,
        features=features,
        obs_window=10,
        max_episode_steps=10,
    )

    assert result["n_steps"] == 10
    assert result["completed_data_split"] is False
    assert result["episode_end_reason"] == "max_episode_steps"
    assert result["survived_full_year"] is False


def test_evaluate_model_uses_requested_observation_schema(monkeypatch):
    """Training and evaluation must construct the same account observation."""
    from decisioning.observations import OBSERVATION_SCHEMA_V2, V2_COORDINATES

    from simulation.environment.env import TradingEnv as OriginalTradingEnv

    env_kwargs = {}

    def _capture_env_kwargs(**kwargs):
        env_kwargs.update(kwargs)
        return OriginalTradingEnv(**kwargs)

    monkeypatch.setattr("evaluation.rollout.model.TradingEnv", _capture_env_kwargs)

    class _SchemaCheckingModel(_ScriptedModel):
        def predict(self, obs, deterministic: bool = True):
            assert np.asarray(obs["account"]).shape[-1] == len(V2_COORDINATES)
            return super().predict(obs, deterministic=deterministic)

    bars = _make_bars(40)
    features = _make_features(bars)
    result = evaluate_model(
        _SchemaCheckingModel({}, default=0),
        bars=bars,
        features=features,
        obs_window=10,
        observation_schema_version=OBSERVATION_SCHEMA_V2,
        observation_age_scale_bars=30.0,
        sl_mode="trailing",
        tp_mode="breakeven",
        tp_reference="prior_session_extreme",
        sl_mode_defer_threshold=0.23,
        tp_mode_defer_threshold=0.29,
        tp_breakeven_alpha=0.37,
    )

    assert result["n_steps"] == len(bars) - 10
    assert env_kwargs["observation_schema_version"] == OBSERVATION_SCHEMA_V2
    assert env_kwargs["observation_age_scale_bars"] == 30.0
    assert env_kwargs["sl_mode"] == "trailing"
    assert env_kwargs["tp_mode"] == "breakeven"
    assert env_kwargs["tp_reference"] == "prior_session_extreme"
    assert env_kwargs["sl_mode_defer_threshold"] == 0.23
    assert env_kwargs["tp_mode_defer_threshold"] == 0.29
    assert env_kwargs["tp_breakeven_alpha"] == 0.37


def test_eval_mode_resumes_trading_after_breach_in_next_session():
    """A guardrail breach in one session must block further trading for the
    rest of that session, but not the entire rollout: the next session_id
    (calendar day) must resume trading normally.

    Regression for episodic=True breach handling ending the whole
    evaluation early (the root cause of the flat-equity/zero-trade bug).
    """
    obs_window = 10
    n = 100
    session_split = 50
    bars = _make_bars(n, session_split=session_split)
    features = _make_features(bars)

    # t=0 -> enter long (bar=obs_window); t=1 -> exit (realises a small
    # spread-only loss); every other call retries entering long.
    script = {0: 1, 1: 19}
    model = _ScriptedModel(script, default=1)

    result = evaluate_model(
        model,
        bars=bars,
        features=features,
        obs_window=obs_window,
        initial_balance=100_000.0,
        # Any realised loss (even just the bid/ask spread) breaches the
        # daily limit immediately, independent of price action.
        guardrail_kwargs={"daily_loss_limit": 0.0001, "max_loss_limit": 1e9},
    )

    trades = result["trades"]
    opens = trades[trades["type"] == "open"]
    assert len(opens) == 2, f"expected exactly 2 opens (session 0 + session 1), got {len(opens)}"
    assert opens.iloc[0]["bar"] == obs_window
    assert opens.iloc[1]["bar"] >= session_split, (
        "second open must occur in session 1, after the breach in session 0 clears"
    )

    # The first session's realised loss blocks the remaining entries for that
    # session. Trading resumes in session 1, whose terminal close also breaches
    # the deliberately tiny per-session loss limit.
    assert result["n_breach_sessions"] == 2
    assert [event["session_id"] for event in result["breach_events"]] == [0, 1]
