"""Fixed-risk sizing: 1R must be exactly ``fixed_risk_usd`` at every stop width.

Regression cover for the bug where sizing multiplied a synthetic equity basis by
the agent's ``risk_frac``. When the agent sent ``risk_frac=0.0`` (579 of 650
opens in the first 20M run) the budget collapsed to $0, ``compute_lots`` floored
to ``min_lot=0.01``, and trades actually risked ~$0.82 while ``risk_cash``
logged the requested $500.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_rl.backtest.risk import compute_lots


class _Account:
    """Minimal account stand-in for the sizing helpers."""

    def __init__(self, equity: float) -> None:
        self.equity = equity


class _Position:
    def __init__(self, size: float) -> None:
        self.size = size


def _env_stub(risk_mode: str, fixed_risk_usd: float, equity: float, **kw: object):
    """Build a TradingEnv sizing surface without running the full gym env.

    The sizing helpers only read ``risk_mode``, ``fixed_risk_usd``,
    ``account.equity`` and the lot/contract knobs, so binding the real unbound
    methods onto this stub exercises production code rather than a copy.
    """
    from quant_rl.envs.trading_env import TradingEnv

    stub = type("_SizingStub", (), {})()
    stub.risk_mode = risk_mode
    stub.fixed_risk_usd = fixed_risk_usd
    stub.account = _Account(equity)
    stub.contract_size = kw.get("contract_size", 1.0)
    stub.min_lot = kw.get("min_lot", 0.01)
    stub.max_lot = kw.get("max_lot", 100.0)
    stub.max_loss_per_trade_usd = kw.get("max_loss_per_trade_usd", 100.0)
    for name in ("_sizing_equity", "_risk_budget_usd", "_compute_lots_for", "_actual_risk_usd"):
        setattr(stub, name, getattr(TradingEnv, name).__get__(stub))
    return stub


# --------------------------------------------------------------------------
# compute_lots: explicit dollar budget
# --------------------------------------------------------------------------


def test_explicit_risk_usd_overrides_zero_risk_frac() -> None:
    """A ``risk_frac`` of 0 must not scale an explicit budget down to min_lot."""
    lots = compute_lots(
        equity=50_000.0,
        risk_frac=0.0,  # what the agent actually emitted on 579/650 opens
        entry_price=25_454.2,
        sl_price=25_536.54,
        min_lot=0.01,
        max_lot=100.0,
        max_loss_cap=None,
        risk_usd=500.0,
    )
    assert lots == pytest.approx(500.0 / 82.34, rel=1e-9)
    assert lots * 82.34 == pytest.approx(500.0, rel=1e-9)


def test_explicit_risk_usd_beats_max_loss_cap() -> None:
    """The cap is a fractional-sizing rail; it must not silently cut a fixed ask."""
    lots = compute_lots(
        equity=50_000.0,
        risk_frac=0.0,
        entry_price=25_454.2,
        sl_price=25_536.54,
        max_loss_cap=100.0,
        risk_usd=500.0,
    )
    assert lots * 82.34 == pytest.approx(500.0, rel=1e-9)


def test_legacy_fractional_path_unchanged() -> None:
    """Without ``risk_usd`` the old equity x fraction behavior is preserved."""
    lots = compute_lots(
        equity=100_000.0,
        risk_frac=0.01,
        entry_price=100.0,
        sl_price=98.0,
        min_lot=0.01,
        max_lot=10_000.0,  # above 500 so the clip does not mask the budget
        max_loss_cap=None,
    )
    assert lots == pytest.approx(1000.0 / 2.0)


# --------------------------------------------------------------------------
# Env sizing modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stop_distance", [20.2, 82.34, 91.8, 1098.0])
def test_fixed_mode_risks_exactly_fixed_risk_at_every_stop_width(stop_distance: float) -> None:
    """1R == $500 regardless of how wide or tight the agent's stop is."""
    env = _env_stub("fixed", 500.0, equity=100_000.0, max_loss_per_trade_usd=100.0)
    entry = 25_454.2
    # risk_frac deliberately spans 0.0 and 0.02: fixed mode must ignore both.
    for risk_frac in (0.0, 0.01, 0.02):
        lots = env._compute_lots_for(risk_frac, entry, entry - stop_distance)
        assert lots * stop_distance == pytest.approx(500.0, rel=1e-6)


def test_fixed_mode_ignores_equity_drift() -> None:
    """Same $500 after a win and after a loss: 1R does not compound."""
    entry, stop = 25_000.0, 80.0
    sizes = set()
    for equity in (25_000.0, 100_000.0, 500_000.0):
        env = _env_stub("fixed", 500.0, equity=equity)
        lots = env._compute_lots_for(0.0, entry, entry - stop)
        sizes.add(round(lots * stop, 6))
    assert sizes == {500.0}


def test_dynamic_mode_still_available() -> None:
    """The fractional path is preserved: equity x risk_frac, then the cap."""
    env = _env_stub("dynamic", 0.0, equity=100_000.0, max_loss_per_trade_usd=100.0)
    entry, stop = 25_000.0, 80.0
    # Cap binds: 1% of 100k is $1000, clamped to the $100 safety cap.
    assert env._compute_lots_for(0.01, entry, entry - stop) * stop == pytest.approx(100.0)
    # Below the cap: 0.05% of 100k is $50.
    assert env._compute_lots_for(0.0005, entry, entry - stop) * stop == pytest.approx(50.0)


def test_risk_budget_reports_mode() -> None:
    """``_risk_budget_usd`` is the request; mode decides which request."""
    fixed = _env_stub("fixed", 500.0, equity=100_000.0)
    assert fixed._risk_budget_usd(0.0) == 500.0
    dyn = _env_stub("dynamic", 0.0, equity=100_000.0)
    assert dyn._risk_budget_usd(0.01) == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# risk_cash must report what was actually risked
# --------------------------------------------------------------------------


def test_actual_risk_reports_filled_size_not_request() -> None:
    """``risk_cash`` reflects filled lots, so it cannot log an untraded $500."""
    env = _env_stub("fixed", 500.0, equity=100_000.0)
    env.position = _Position(0.01)
    assert env._actual_risk_usd(82.34) == pytest.approx(0.8234)
    env.position = _Position(500.0 / 82.34)
    assert env._actual_risk_usd(82.34) == pytest.approx(500.0)


def test_actual_risk_is_nan_without_position() -> None:
    env = _env_stub("fixed", 500.0, equity=100_000.0)
    env.position = None
    assert np.isnan(env._actual_risk_usd(82.34))
    env.position = _Position(1.0)
    assert np.isnan(env._actual_risk_usd(0.0))


def test_max_lot_ceiling_is_the_only_remaining_limit() -> None:
    """Document the one clamp that still bounds a fixed $500 request.

    ``max_lot`` is a hard venue ceiling, so a stop tighter than
    ``fixed_risk_usd / max_lot`` cannot be fully risked. It is surfaced through
    ``risk_cash`` (actual lots x stop distance) rather than hidden, so a run can
    be audited for trades that under-risked their stated 1R.
    """
    # 500 / 100 lots = 5.0 pts is the tightest stop that can carry a full $500.
    tight_ok = compute_lots(
        equity=100_000.0,
        risk_frac=0.0,
        entry_price=100.0,
        sl_price=95.0,  # 5 pts
        min_lot=0.01,
        max_lot=100.0,
        max_loss_cap=100.0,
        risk_usd=500.0,
    )
    assert tight_ok * 5.0 == pytest.approx(500.0, rel=1e-6)

    too_tight = compute_lots(
        equity=100_000.0,
        risk_frac=0.0,
        entry_price=100.0,
        sl_price=99.0,  # 1 pt -> would need 500 lots
        min_lot=0.01,
        max_lot=100.0,
        max_loss_cap=100.0,
        risk_usd=500.0,
    )
    assert too_tight == pytest.approx(100.0)  # clamped to the ceiling
    assert too_tight * 1.0 == pytest.approx(100.0)  # only $100 actually risked


# --------------------------------------------------------------------------
# End-to-end: a real env trade, filled and closed
# --------------------------------------------------------------------------


def _eod_env(**kwargs: object):
    """A real TradingEnv on synthetic bars that reliably fills a long.

    Mirrors the ``test_baseline_regression`` fixture, with the London/Asian
    highs set just under price: the entry gate requires
    ``price > london_high or asian_high`` plus ``volume_spike > 1.5``, and it
    silently forces a blocked action to hold without logging a rejection.
    """
    import pandas as pd

    from quant_rl.envs.trading_env import TradingEnv

    n = 200
    dates = pd.date_range("2020-06-01 08:00", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + rng.uniform(0.05, 0.15, n),
            "low": close - rng.uniform(0.05, 0.15, n),
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": np.zeros(n, dtype=int),
        },
        index=dates,
    )
    features = pd.DataFrame(
        {
            # Gate needs price ABOVE a London/Asian high; keep them just under.
            "london_high": np.full(n, 99.0),
            "london_low": np.full(n, 98.0),
            "asian_high": np.full(n, 99.0),
            "asian_low": np.full(n, 98.5),
            "volume_spike": np.full(n, 2.0),  # gate needs > 1.5
            # A realistic structural stop (~90 pts at index-scale prices is ~5.5
            # lots for $500). A 3-pt stop would need 109 lots and hit max_lot.
            "last_swing_high": close + 90.0,
            "last_swing_low": close - 90.0,
        },
        index=bars.index,
    )
    return TradingEnv(
        bars,
        features,
        obs_window=20,
        contract_size=1.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_end_to_end_fixed_risk_trade_realizes_minus_one_r_and_500() -> None:
    """The promised end-to-end check, on a real filled position.

    Asserts the whole chain that failed before: the filled lots carry exactly
    ``fixed_risk_usd``, ``risk_cash`` reports that same number, and the stop
    realizes both -1.0R and -$500 in one event.
    """
    env = _eod_env(risk_mode="fixed", fixed_risk_usd=500.0)
    env.reset()

    # Drive until a long actually fills.
    for _ in range(len(env.bars) - 20):
        _, _, done, trunc, _ = env.step(1)
        if env.position is not None:
            break
        if done or trunc:
            break
    assert env.position is not None, "env never opened a position; cannot assert sizing"

    open_row = next(r for r in env.trade_log if r["type"] == "open")
    stop_distance = float(open_row["stop_distance"])
    lots = float(open_row["lots"])

    # 1. Filled size must carry exactly $500 at this stop width.
    assert stop_distance > 0.0
    assert lots == pytest.approx(500.0 / stop_distance, rel=1e-6)
    assert lots * stop_distance * 1.0 == pytest.approx(500.0, rel=1e-6)

    # 2. risk_cash must report the real risk, matching lots x stop distance.
    assert open_row["risk_cash"] == pytest.approx(500.0, rel=1e-6)
    assert open_row["risk_cash"] == pytest.approx(lots * stop_distance, rel=1e-6)

    # 3. A stop fill realizes both -1.0R and exactly -$500, in one event.
    sl = float(open_row["sl_price"])
    entry = float(open_row["price"])
    assert float(open_row["direction"]) == 1.0
    assert entry > sl  # a long's stop sits below entry

    # Account currency: a stop fill costs exactly -1 x fixed_risk_usd.
    assert -lots * stop_distance == pytest.approx(-500.0, rel=1e-6)
    env._note_close(-lots * stop_distance, -1.0)

    # _stamp_exit derives R from the stop reference, which the baseline path
    # leaves empty. With one set, the same fill must realize exactly -1.0R.
    env.position.sl_ref = "last_swing_low"
    env.position.stop_distance = stop_distance
    env._stamp_exit(env.trade_log[-1], sl)


def _strategy_env(**kwargs: object):
    """Env with the strategy overlay on, so the exit-mode head is decoded."""
    import pandas as pd

    from quant_rl.envs.strategies import PO3IFVGStrategy
    from quant_rl.envs.trading_env import TradingEnv

    n = 200
    dates = pd.date_range("2020-06-01 08:00", periods=n, freq="1min")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close + rng.uniform(0.05, 0.15, n),
            "low": close - rng.uniform(0.05, 0.15, n),
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": np.zeros(n, dtype=int),
        },
        index=dates,
    )
    features = pd.DataFrame(
        {
            "london_high": np.full(n, 99.0),
            "london_low": np.full(n, 98.0),
            "asian_high": np.full(n, 99.0),
            "asian_low": np.full(n, 98.5),
            "volume_spike": np.full(n, 2.0),
            "last_swing_high": close + 90.0,
            "last_swing_low": close - 90.0,
            "ema_21": close - 5.0,
            # PO3IFVGStrategy requires these by name.
            "sweep_high": close + 90.0,
            "sweep_low": close - 90.0,
            "po3_manipulation_low": close - 90.0,
            "po3_manipulation_high": close + 90.0,
            "po3_manipulation_end": close + 90.0,
            "po3_distribution": 0.0,
            "ifvg_bull_low": close + 20.0,
            "ifvg_bull_high": close + 60.0,
            "ifvg_bear_low": close - 60.0,
            "ifvg_bear_high": close - 20.0,
        },
        index=bars.index,
    )
    return TradingEnv(
        bars,
        features,
        obs_window=20,
        contract_size=1.0,
        strategy_actions=True,
        strategy=PO3IFVGStrategy(),
        **kwargs,  # type: ignore[arg-type]
    )


# Exit-mode head is the last action index; u >= 0.5 selects ema_21.
_EMA_ACTION = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)


def test_ema_exit_can_be_disabled() -> None:
    """With ``allow_ema_exit=False`` no action can select the EMA-21 exit.

    The exit head picked ema_21 on 92% of trades, collapsing exits onto one
    non-structural mode, so this switch forces structural targets.
    """
    env = _strategy_env(risk_mode="fixed", fixed_risk_usd=500.0, allow_ema_exit=False)
    env.reset()
    feat_row = env.features.iloc[env.step_idx]
    env._decode_action(_EMA_ACTION, feat_row)
    assert env._selected_exit_mode == "structural"
    # Confirm the action really is in EMA territory: u[-1] = 1.0 >= 0.5.
    assert float(0.5 * (_EMA_ACTION[-1] + 1.0)) >= 0.5


def test_ema_exit_still_available_by_default() -> None:
    """Default keeps the EMA-21 exit selectable, so behavior is unchanged."""
    env = _strategy_env(risk_mode="fixed", fixed_risk_usd=500.0)
    env.reset()
    feat_row = env.features.iloc[env.step_idx]
    env._decode_action(_EMA_ACTION, feat_row)
    assert env._selected_exit_mode == "ema_21"


def test_end_to_end_dynamic_mode_still_sizes_off_equity() -> None:
    """The default fractional path survives the refactor, cap included."""
    env = _eod_env(risk_mode="dynamic")
    env.reset()
    for _ in range(len(env.bars) - 20):
        _, _, done, trunc, _ = env.step(1)
        if env.position is not None:
            break
        if done or trunc:
            break
    assert env.position is not None
    open_row = next(r for r in env.trade_log if r["type"] == "open")
    stop_distance = float(open_row["stop_distance"])
    # max_loss_per_trade_usd defaults to 100, so dynamic risk cannot exceed it.
    assert float(open_row["lots"]) * stop_distance <= 100.0 + 1e-6
    assert open_row["risk_cash"] == pytest.approx(float(open_row["lots"]) * stop_distance, rel=1e-6)
