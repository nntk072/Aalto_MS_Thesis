"""Roll a trained PPO model through :class:`TradingEnv` to produce trades.

``quant_rl.backtest.engine.run_backtest`` drives a rule-based ``policy``
callable (plain ``np.ndarray`` in, ``int`` action out in ``{-1, 0, 1, exit}``)
through a lightweight event loop. The RL model, however, was trained against
``TradingEnv``'s own **Dict** observation (``{"seq": ..., "account": ...}``)
and **Discrete(20)** action space (which additionally encodes risk_frac/
rr_ratio variants for entries) — those formats don't match
``run_backtest``'s interface at all.

Rather than reverse-engineer an adapter, this module evaluates the model by
walking it through the *exact* environment class/action space it was trained
on, using ``TradingEnv(..., episodic=False)`` so a guardrail breach blocks
new trading for the rest of that session instead of ending the whole
rollout early (see the ``episodic`` parameter docstring on ``TradingEnv``).

The returned dict mirrors ``run_backtest``'s return shape so it can be
passed straight into ``quant_rl.eval.metrics.calculate_metrics`` and
``quant_rl.eval.export.save_run`` unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..backtest.costs import COST_US100, CostModel
from ..envs.trading_env import TradingEnv


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
) -> dict[str, Any]:
    """Walk a trained PPO ``model`` over *bars*/*features* and collect trades.

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
    )

    obs, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        raw_action, _ = model.predict(obs, deterministic=deterministic)
        action = int(np.asarray(raw_action).reshape(-1)[0])
        obs, _, done, truncated, _ = env.step(action)

    equity_series = pd.Series(env.equity_curve[1:], index=env.bars.index[env.obs_window :])
    trades_df = pd.DataFrame(env.trade_log)

    return {
        "equity": equity_series,
        "trades": trades_df,
        "account": env.account,
        "breaches": env.breach_log,
        "breach_events": env.breach_events,
        "n_sessions": len(env.all_sessions),
        "n_breach_sessions": len(env.breached_sessions),
        "n_sessions_with_trades": len(env.sessions_with_trades),
        "n_sessions_skipped": len(env.breached_sessions),
    }
