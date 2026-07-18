"""Tests for quant_rl.eval.rollout.evaluate_model (RL model walk-forward eval).

Regression coverage for the bug where RL runs reported ``equity == 100k``
for every case with zero trades executed: the evaluation step used a
hardcoded hold-forever stub instead of querying the trained model. These
tests exercise the real fix — rolling a model through
``TradingEnv(episodic=False)`` — with a scripted stand-in model so no actual
PPO training is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.eval.rollout import evaluate_model


class _ScriptedModel:
    """Minimal stand-in for a trained SB3 model's ``.predict()`` API.

    Ignores the observation and returns a scripted action per call index,
    mirroring ``model.predict(obs, deterministic=True) -> (action, state)``.
    """

    def __init__(self, script: dict[int, int], default: int = 0):
        self.script = script
        self.default = default
        self.t = 0

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
    """Trivial feature matrix aligned 1:1 with *bars* (no swing/SMT columns
    needed — entries fall back to lots=1.0, sl_price/tp_price=None)."""
    return pd.DataFrame({"ret_1": np.zeros(len(bars))}, index=bars.index)


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
    assert len(equity) == n - obs_window
    assert isinstance(equity.index, pd.DatetimeIndex)

    close_rows = trades[trades["type"] == "close"]
    assert not close_rows.empty
    assert close_rows["price"].notna().all()
    assert close_rows["bar"].notna().all()
    # Equity must not stay flat at the initial balance the whole rollout.
    assert not np.isclose(equity.iloc[-1], 100_000.0)


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

    assert result["n_breach_sessions"] == 1
    assert len(result["breach_events"]) == 1
    assert result["breach_events"][0]["session_id"] == 0
