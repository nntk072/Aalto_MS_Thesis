"""Funnel accounting for strategy entry refusals (SL/TP v2 phase 1).

Every site that refuses an entry must go through ``TradingEnv._reject`` with a
reason registered in ``_REJECTION_COUNTERS``. These tests pin the funnel:
counter and reason agree, ``entry_rejected_total`` always equals the sum of the
``rejected_*`` buckets, and ``reset`` clears the episode's books.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_rl.envs.strategies import PO3IFVGStrategy
from quant_rl.envs.trading_env import (
    _REJECTION_COUNTERS,
    TradingEnv,
)


def _bars(n: int = 20, spread_pts: float | None = None) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01 16:30", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    close = np.clip(close, 99.0, 101.0)
    df = pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "session_id": np.zeros(n, dtype=int),
            "session": "ny",
        },
        index=idx,
    )
    if spread_pts is not None:
        # Raw MT5 broker-points column; the env scales it by point_size.
        df["spread"] = float(spread_pts)
    return df


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
            "po3_manipulation_end": np.ones(n),
            "po3_distribution": np.zeros(n),
            "po3_distribution_direction": np.zeros(n),
            "ifvg_bull_low": np.full(n, 98.0),
            "ifvg_bull_high": np.full(n, 99.0),
            "price_in_ifvg_bull": np.ones(n) if is_long else np.zeros(n),
            "price_in_ifvg_bear": np.zeros(n) if is_long else np.ones(n),
            "ifvg_retest_bull": np.ones(n) if is_long else np.zeros(n),
            "ifvg_retest_bear": np.zeros(n) if is_long else np.ones(n),
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
            "prev_day_high": np.full(n, 150.0),
            "prev_day_low": np.full(n, 50.0),
        },
        index=bars.index,
    )


def _policy_env(
    feats: pd.DataFrame,
    bars: pd.DataFrame,
    strategy: Any | None = None,
    **kwargs: Any,
) -> TradingEnv:
    env_kwargs: dict[str, Any] = {
        "strategy_actions": True,
        "strategy": strategy or PO3IFVGStrategy(enforce_gate=False),
        "obs_window": 5,
        "risk_frac_range": (0.01, 0.01),
        "min_sl_points": 0.0,
        "min_sl_atr_mult": 0.0,
        "open_manipulation_bars": 0,
        "entry_cooldown_bars": 0,
        "block_overnight": False,
        "max_episode_steps": None,
    }
    env_kwargs.update(kwargs)
    return TradingEnv(bars, feats, **env_kwargs)


def test_registry_covers_every_counter() -> None:
    """Each rejected_* bucket in _empty_entry_diag has exactly one reason key."""
    from quant_rl.envs.trading_env import _empty_entry_diag

    buckets = {k for k in _empty_entry_diag(entry_state_machine=True) if k.startswith("rejected_")}
    assert set(_REJECTION_COUNTERS.values()) == buckets


def test_rejected_total_tracks_the_buckets() -> None:
    bars = _bars()
    feats = _features(bars)
    env = _policy_env(feats, bars, max_entries_per_session=1)
    env.reset()
    total = 0
    for _ in range(12):
        _obs, _reward, _done, _trunc, info = env.step(
            np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
        )
        bucket_sum = sum(v for k, v in env._entry_diag.items() if k.startswith("rejected_"))
        assert info["entry_rejected_total"] == bucket_sum
        total = bucket_sum
    # At most one open per session: later attempts land in rejected_session_cap.
    assert total >= 1
    assert info["entry_rejected_session_cap"] >= 1


def test_session_cap_counts_and_names_the_reason() -> None:
    bars = _bars()
    feats = _features(bars)
    env = _policy_env(feats, bars, max_entries_per_session=1)
    env.reset()
    reasons: list[str] = []
    for _ in range(8):
        _obs, _reward, _done, _trunc, info = env.step(
            np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
        )
        reasons.append(str(info["entry_rejection_reason"]))
    assert "session_cap" in reasons
    assert info["entry_rejected_session_cap"] >= 1
    assert info["entry_opened"] == 1


def test_cooldown_counts_and_names_the_reason() -> None:
    bars = _bars(n=40)
    feats = _features(bars)
    env = _policy_env(feats, bars, entry_cooldown_bars=100)
    env.reset()
    # Arm the post-close cooldown directly: _note_close sets this after a real
    # close, and the gate under test is the refusal, not the close itself.
    env._cooldown_until_bar = 10**9
    reasons: list[str] = []
    for _ in range(10):
        _obs, _reward, _done, _trunc, info = env.step(
            np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
        )
        reasons.append(str(info["entry_rejection_reason"]))
    assert "entry_cooldown" in reasons
    assert info["entry_rejected_cooldown"] >= 1


def test_gate_rejection_counts_and_names_the_reason() -> None:
    bars = _bars()
    feats = _features(bars)
    env = _policy_env(feats, bars)
    env.reset()
    # Refuse every side: the gate branch under test is the env's counter and
    # reason stamping, not any one strategy's setup logic (covered elsewhere).
    env.strategy.validate_entry = lambda *, direction, row: False  # type: ignore[method-assign]
    _obs, _reward, _done, _trunc, info = env.step(
        np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
    )
    assert env.position is None
    assert info["entry_rejection_reason"] == "entry_gate"
    assert info["entry_rejected_gate"] == 1


def test_rr_window_prefers_in_range_targets() -> None:
    from quant_rl.backtest.entry_levels import _target_menu

    row = pd.Series(
        {
            "last_swing_high": 101.0,  # 1.0 dist on a 1.0 stop -> exactly 1R
            "asian_high": 110.0,  # 10R: legal but far
            "prev_day_high": 105.0,  # 5R: inside the wide window
        }
    )
    wide = _target_menu(
        direction=1,
        entry_price=100.0,
        row=row,
        stop_dist=1.0,
        max_tp_distance=float("nan"),
        rr_bounds=(0.5, 20.0),
    )
    assert [name for name, _, _ in wide] == ["last_swing_high", "prev_day_high", "asian_high"]
    # Tight window: only the 1R level qualifies, so the menu shrinks to it.
    tight = _target_menu(
        direction=1,
        entry_price=100.0,
        row=row,
        stop_dist=1.0,
        max_tp_distance=float("nan"),
        rr_bounds=(0.9, 1.1),
    )
    assert [name for name, _, _ in tight] == ["last_swing_high"]


def test_rr_window_falls_back_when_nothing_qualifies() -> None:
    from quant_rl.backtest.entry_levels import _target_menu

    row = pd.Series({"last_swing_high": 101.0, "asian_high": 110.0})
    menu = _target_menu(
        direction=1,
        entry_price=100.0,
        row=row,
        stop_dist=1.0,
        max_tp_distance=float("nan"),
        rr_bounds=(2.0, 3.0),
    )
    # 1R and 10R both miss the window, but the menu stays usable.
    assert [name for name, _, _ in menu] == ["last_swing_high", "asian_high"]


def test_stop_buffer_honors_config() -> None:
    from quant_rl.backtest.entry_levels import stop_buffer

    # 0.0 is exact; an explicit buffer wins over any ATR heuristic.
    assert stop_buffer(0.0) == 0.0
    assert stop_buffer(2.5) == 2.5
    assert stop_buffer(-1.0) == 0.0


def test_sl_already_hit_counts_and_names_the_reason() -> None:
    # Dip the fill-bar close to 97 while the decision bar is still at 100: the
    # 98 structural stop then sits below the ask entry (117, lifted by the
    # 20-point bar spread) but above the fill bid (97), i.e. already hit.
    bars = _bars(spread_pts=2000.0)
    closes = np.asarray(bars["close"])
    closes[1:] = 97.0
    bars["close"] = closes
    bars["high"] = closes + 0.05
    bars["low"] = closes - 0.05
    feats = _features(bars)
    env = _policy_env(feats, bars)
    env.reset()
    reasons: list[str] = []
    for _ in range(6):
        _obs, _reward, _done, _trunc, info = env.step(
            np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32)
        )
        reasons.append(str(info["entry_rejection_reason"]))
    assert env.position is None
    assert "sl_already_hit" in reasons
    assert info["entry_rejected_sl_already_hit"] >= 1


def test_reset_clears_the_funnel() -> None:
    bars = _bars()
    feats = _features(bars)
    env = _policy_env(feats, bars, max_entries_per_session=1)
    env.reset()
    for _ in range(8):
        env.step(np.array([0.9, 0.0, 0.5, 0.0, -1.0], dtype=np.float32))
    assert sum(env._entry_diag.values()) > 0
    env.reset()
    assert sum(env._entry_diag.values()) == 0


def test_config_slippage_reaches_sl_tp_fills() -> None:
    """``costs.slippage_points`` (price points) widens stop/target fills.

    Pins the wiring ``default.yaml -> _cost_model -> TradingEnv.cost_model ->
    _sl_tp_fill_quote``: a 5-point slippage must worsen both sides by exactly
    5 points, and ``min_sl_points=10`` keeps its 2x sizing margin over it.
    """
    from omegaconf import OmegaConf

    from quant_rl.backtest.costs import CostModel
    from quant_rl.config import load_config
    from quant_rl.train.train_rl import _cost_model

    cfg = load_config([])
    assert float(cfg.costs.slippage_points) == 5.0
    assert float(cfg.risk.min_sl_points) == 10.0
    assert float(cfg.risk.min_sl_points) >= 2 * float(cfg.costs.slippage_points)

    model = _cost_model(cfg)
    assert isinstance(model, CostModel)
    assert model.slippage_points == 5.0

    bars = _bars()
    feats = _features(bars)
    env = _policy_env(feats, bars, cost_model=model)
    assert env.cost_model.slippage_points == 5.0

    bid, ask = env._sl_tp_fill_quote(direction=1, hit_price=100.0)
    assert bid == 100.0 - 5.0
    assert ask == bid + model.spread_points * model.point_size
    bid_s, ask_s = env._sl_tp_fill_quote(direction=-1, hit_price=100.0)
    assert ask_s == 100.0 + 5.0
    assert bid_s == ask_s - model.spread_points * model.point_size

    us500_cfg = OmegaConf.merge(cfg, {"data": {"primary": "US500.cash"}})
    assert _cost_model(us500_cfg).spread_points == float(cfg.costs.spread_us500)
