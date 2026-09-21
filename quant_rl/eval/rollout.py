"""Roll a trained PPO/SAC model through :class:`TradingEnv` to produce trades.

``quant_rl.backtest.engine.run_backtest`` drives a rule-based ``policy``
callable (plain ``np.ndarray`` in, ``int`` action out in ``{-1, 0, 1, exit}``)
through a lightweight event loop. The RL model, however, was trained against
``TradingEnv``'s own **Dict** observation (``{"seq": ..., "account": ...}``)
and **Discrete(20)** action space (which additionally encodes risk_frac/
rr_ratio variants for entries) or **Box(-1, 1)** for continuous actions —
those formats don't match ``run_backtest``'s interface at all.

Rather than reverse-engineer an adapter, this module evaluates the model by
walking it through the *exact* environment class/action space it was trained
on, using ``TradingEnv(..., episodic=False)`` so a guardrail breach blocks
new trading for the rest of that session instead of ending the whole
rollout early (see the ``episodic`` parameter docstring on ``TradingEnv``).

The returned dict mirrors ``run_backtest``'s return shape so it can be
passed straight into ``quant_rl.evaluation.calculate_metrics`` and
``quant_rl.eval.export.save_run`` unchanged.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.costs import COST_US100, CostModel
from ..envs.trading_env import TradingEnv
from ..evaluation.metrics import max_drawdown


def make_action_fn(
    model: Any,
    continuous_actions: bool = False,
    deterministic: bool = True,
    strategy_actions: bool = False,
) -> Any:
    """Return an observation → action callable with explicit action typing.

    PPO trains on ``TradingEnv``'s discrete action space, so the policy's
    integer action id is passed through unchanged; SAC trains on the
    continuous Box space and yields a float position-sizing fraction.
    When ``strategy_actions`` is set (Idea 1 PO3 Box(4)), the full float32
    vector is returned so eval matches training.

    This is the single shared implementation — ``scripts/train_rl.py`` and
    ``scripts/compare_encoders.py`` used to each carry private copies.
    """

    def action_fn(obs: dict[str, Any]) -> Any:
        action = model.predict(obs, deterministic=deterministic)[0]
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if strategy_actions:
            return arr.astype(np.float32, copy=False)
        scalar = arr[0]
        return float(scalar) if continuous_actions else int(scalar)

    return action_fn


def evaluate_model(
    model: Any,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    obs_window: int = 60,
    initial_balance: float = 100_000.0,
    cost_model: CostModel = COST_US100,
    broker_kwargs: dict[str, Any] | None = None,
    guardrail_kwargs: dict[str, Any] | None = None,
    risk_frac_range: tuple[float, float] = (0.005, 0.02),
    rr_ratio_range: tuple[float, float] = (1.0, 3.0),
    swing_buffer_pts: float = 1.0,
    min_lot: float = 0.01,
    max_lot: float = 100.0,
    contract_size: float = 1.0,
    max_loss_per_trade_usd: float = 100.0,
    dsr_eta: float = 0.01,
    deterministic: bool = True,
    max_episode_steps: int | None = None,
    continuous_actions: bool = False,
    use_sweep_reward: bool = False,
    sweep_alpha: float = 0.1,
    sweep_beta: float = 0.01,
    sweep_hold_bars: int = 3,
    dsr_weight: float = 0.3,
    sweep_weight: float = 0.7,
    block_overnight: bool = True,
    eod_risk: dict[str, Any] | None = None,
    use_vae: bool = False,
    vae: Any | None = None,
    strategy: Any = None,
    strategy_actions: bool = False,
    strategy_reward: Any = None,
    strategy_weight: float = 0.0,
    sl_buffer_pts: float = 0.0,
    pre_ny_by_date: dict[Any, Any] | None = None,
    min_sl_atr_mult: float = 0.5,
    min_sl_points: float = 0.0,
    max_entries_per_session: int = 0,
    entry_cooldown_bars: int = 0,
    reward_mode: str = "dsr",
    entry_intensity_threshold: float = 0.1,
) -> dict[str, Any]:
    """Walk a trained RL ``model`` over *bars*/*features* and collect trades.

    Parameters mirror ``TradingEnv.__init__`` exactly. ``risk_frac_range`` and
    ``rr_ratio_range`` change what the model's discrete entry actions *mean*
    (they define the 3x3 grid of risk/reward variants), so callers **must**
    pass the same ranges used to build the training environment or the
    model's learned action semantics will not line up.

    Returns
    -------
    dict with keys ``equity``, ``trades``, ``account``, ``breaches``,
    ``breach_events``, ``n_sessions``, ``n_breach_sessions``,
    ``n_sessions_with_trades``, ``n_sessions_skipped`` — the same shape
    produced by ``quant_rl.backtest.engine.run_backtest``.
    """
    env = TradingEnv(
        bars=bars,
        features=features,
        obs_window=obs_window,
        initial_balance=initial_balance,
        cost_model=cost_model,
        broker_kwargs=broker_kwargs,
        guardrail_kwargs=guardrail_kwargs,
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=swing_buffer_pts,
        min_lot=min_lot,
        max_lot=max_lot,
        contract_size=contract_size,
        max_loss_per_trade_usd=max_loss_per_trade_usd,
        dsr_eta=dsr_eta,
        episodic=False,
        max_episode_steps=max_episode_steps,
        continuous_actions=continuous_actions,
        use_sweep_reward=use_sweep_reward,
        sweep_alpha=sweep_alpha,
        sweep_beta=sweep_beta,
        sweep_hold_bars=sweep_hold_bars,
        dsr_weight=dsr_weight,
        sweep_weight=sweep_weight,
        block_overnight=block_overnight,
        eod_risk=eod_risk,
        use_vae=use_vae,
        vae=vae,
        strategy=strategy,
        strategy_actions=strategy_actions,
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        sl_buffer_pts=sl_buffer_pts,
        pre_ny_by_date=pre_ny_by_date,
        min_sl_atr_mult=min_sl_atr_mult,
        min_sl_points=min_sl_points,
        max_entries_per_session=max_entries_per_session,
        entry_cooldown_bars=entry_cooldown_bars,
        reward_mode=reward_mode,
        entry_intensity_threshold=entry_intensity_threshold,
    )

    obs, _ = env.reset()
    done = False
    truncated = False
    action_fn = make_action_fn(
        model, continuous_actions, deterministic, strategy_actions=strategy_actions
    )
    action_counts: Counter[str] = Counter()
    while not (done or truncated):
        action = action_fn(obs)
        if strategy_actions:
            action_key = str(tuple(np.round(np.asarray(action).reshape(-1), 4)))
        else:
            action_key = str(action)
        action_counts[action_key] += 1
        obs, _, done, truncated, _ = env.step(action)

    equity_series = pd.Series(
        env.equity_curve,
        index=pd.DatetimeIndex(env.equity_times),
    )
    trades_df = pd.DataFrame(env.trade_log)

    n_sessions = len(env.all_sessions)
    days_traded = len(env.sessions_with_trades)
    fail_time = env.breach_events[0]["time"] if env.breach_events else None
    survived_full_year = len(env.breach_events) == 0

    return {
        "equity": equity_series,
        "trades": trades_df,
        "account": env.account,
        "breaches": env.breach_log,
        "breach_events": env.breach_events,
        "n_sessions": n_sessions,
        "n_breach_sessions": len(env.breached_sessions),
        "n_sessions_with_trades": days_traded,
        "n_sessions_skipped": n_sessions - days_traded,
        "days_traded": days_traded,
        "survived_full_year": survived_full_year,
        "fail_time": fail_time,
        "max_trailing_dd": float(max_drawdown(env.equity_curve)),
        "action_counts": dict(action_counts),
        "entry_diag": dict(env._entry_diag),
    }
