"""Build simulation environments from resolved domain configuration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_data.loading.ticks import TickBook
from simulation.environment.distribution_reward import DistributionReward
from simulation.environment.env import TradingEnv
from simulation.environment.po3_reward import PO3Reward
from trading.execution.costs import CostModel
from trading.policy.config import resolve_trading_policy
from trading.strategies import (
    BaselineStrategy,
    DistributionStrategy,
    PO3IFVGStrategy,
    TradingStrategy,
)


def strategy_risk_ranges(cfg: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Risk/RR ranges for TradingEnv; repeated-payout owns the risk range."""
    policy = resolve_trading_policy(cfg)
    if policy.repeated_payout:
        rr_block = (
            getattr(cfg, "strategy", {}).get("risk", {})
            if hasattr(getattr(cfg, "strategy", {}), "get")
            else {}
        )
        rr = list(rr_block.get("rr_range", [1.5, 5.0]))
        return (0.0, policy.max_agent_risk_frac), (float(rr[0]), float(rr[1]))
    if bool(cfg.env.get("strategy_actions", False)):
        risk_cfg = getattr(cfg, "strategy", None)
        risk_block = risk_cfg.get("risk", {}) if risk_cfg is not None else {}
        rfr = list(risk_block.get("risk_frac_range", [0.005, 0.01]))
        rr = list(risk_block.get("rr_range", [1.5, 5.0]))
        return (float(rfr[0]), float(rfr[1])), (float(rr[0]), float(rr[1]))
    return (
        (cfg.risk.default_risk_frac * 0.5, cfg.risk.default_risk_frac * 2.0),
        (cfg.risk.rr_ratio_default * 0.5, cfg.risk.rr_ratio_default * 1.5),
    )


def max_loss_per_trade(cfg: Any) -> float:
    """Per-trade USD cap used by legacy sizing; repeated-payout caps dynamically."""
    policy = resolve_trading_policy(cfg)
    if policy.repeated_payout:
        return policy.operating_balance * policy.max_agent_risk_frac
    if bool(cfg.env.get("strategy_actions", False)):
        hard_limit = float(cfg.ftmo.risk_per_trade_limit)
        sizing_limit = float(cfg.ftmo.get("risk_sizing_limit_usd", hard_limit))
        return min(hard_limit, sizing_limit)
    return float(cfg.backtest.validation.max_loss_per_trade_usd)


def guardrail_kwargs(cfg: Any) -> dict[str, float]:
    """Resolve policy-specific hard/soft guardrails."""
    policy = resolve_trading_policy(cfg)
    if policy.repeated_payout:
        return {
            "daily_loss_limit": policy.daily_loss_limit,
            "max_loss_limit": policy.operating_balance - policy.account_floor,
            "risk_per_trade_limit": policy.operating_balance * policy.max_agent_risk_frac,
            "soft_daily_loss_limit": 0.0,
            "soft_max_loss_limit": 0.0,
            "trailing_dd_limit": 0.0,
            "soft_trailing_dd_limit": 0.0,
        }
    return {
        "daily_loss_limit": float(cfg.ftmo.daily_loss_limit),
        "max_loss_limit": float(cfg.ftmo.max_loss_limit),
        "risk_per_trade_limit": float(cfg.ftmo.risk_per_trade_limit),
        "soft_daily_loss_limit": float(cfg.ftmo.get("soft_daily_loss_limit", 2000.0)),
        "soft_max_loss_limit": float(cfg.ftmo.get("soft_max_loss_limit", 5000.0)),
        "trailing_dd_limit": float(cfg.ftmo.get("trailing_dd_limit", 0.07)),
        "soft_trailing_dd_limit": float(cfg.ftmo.get("soft_trailing_dd_limit", 0.0)),
    }


def eval_guardrail_kwargs(cfg: Any) -> dict[str, float]:
    """FTMO rules for train/test rollouts: daily $5k + max $10k from initial."""
    kwargs = guardrail_kwargs(cfg)
    kwargs["trailing_dd_limit"] = 0.0
    kwargs["soft_trailing_dd_limit"] = 0.0
    return kwargs


def max_episode_steps(cfg: Any) -> int | None:
    """Parse ``env.max_episode_steps``; YAML ``null`` → full-year episode."""
    raw = cfg.env.get("max_episode_steps", None)
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("", "null", "none"):
        return None
    return int(raw)


def strategy_from_cfg(cfg: Any) -> tuple[Any, Any, float]:
    """Build (strategy, strategy_reward, strategy_weight) from merged config.

    Baseline (Idea 3, P2) keeps the no-op strategy and no alignment reward
    unless ``env.strategy_actions`` is explicitly enabled by a variant config.
    """
    strat_cfg = getattr(cfg, "strategy", None)
    if strat_cfg is None or not bool(cfg.env.get("strategy_actions", False)):
        return BaselineStrategy(), None, 0.0

    name = str(strat_cfg.get("name", "baseline"))
    manipulation_filter = str(strat_cfg.get("manipulation_filter", "against"))
    enforce_gate = bool(strat_cfg.get("entry", {}).get("enforce_gate", False))
    reward_cfg = strat_cfg.get("reward", {})
    weight = float(reward_cfg.get("strategy_weight", 0.0)) if reward_cfg else 0.0

    reward: PO3Reward | DistributionReward | None = None
    strategy: TradingStrategy
    if name == "po3_ifvg":
        ifvg_cfg = strat_cfg.get("ifvg", {}) or {}
        strategy = PO3IFVGStrategy(
            enforce_gate=enforce_gate,
            require_price_retest=bool(ifvg_cfg.get("require_price_retest", True)),
            entry_mode=str(strat_cfg.get("entry", {}).get("mode", "legacy")),
            candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
            variant_id=str(strat_cfg.get("variant_id", "po3_ifvg_legacy")),
        )
        strategy.manipulation_filter = manipulation_filter
        reward = PO3Reward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            manipulation_penalty=float(reward_cfg.get("manipulation_penalty", 0.02)),
            invalid_ifvg_penalty=float(reward_cfg.get("invalid_ifvg_penalty", 0.01)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
        )
    elif name == "distribution":
        previous_session_pd_gate = bool(
            strat_cfg.get("entry", {}).get("previous_session_pd_gate", False)
        )
        if previous_session_pd_gate and not bool(
            cfg.features.get("include_previous_session_pd", False)
        ):
            raise ValueError(
                "strategy.entry.previous_session_pd_gate requires "
                "features.include_previous_session_pd=true"
            )
        strategy = DistributionStrategy(
            enforce_gate=enforce_gate,
            entry_mode=str(strat_cfg.get("entry", {}).get("mode", "legacy")),
            candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
            previous_session_pd_gate=previous_session_pd_gate,
            variant_id=str(strat_cfg.get("variant_id", "distribution_legacy")),
        )
        strategy.manipulation_filter = manipulation_filter
        reward_cfg = strat_cfg.get("reward", {}) or {}
        reward = DistributionReward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
        )
    else:
        strategy = BaselineStrategy()
        reward = None
        weight = 0.0
    return strategy, reward, weight


def build_eval_common(
    cfg: Any,
    args: Any | None = None,
    env_vae: Any | None = None,
    pre_ny_by_date: dict[Any, Any] | None = None,
    arch: str | None = None,
    tick_fill_slippage_points: float = 0.0,
) -> tuple[dict[str, Any], Any, Any, float, bool]:
    """Kwargs shared by the in-loop dashboard slice and the final eval."""
    strategy, strategy_reward, strategy_weight = strategy_from_cfg(cfg)
    strategy_actions = bool(cfg.env.get("strategy_actions", False))
    sl_buffer_pts = float(cfg.env.get("sl_buffer_pts", 0.0))
    risk_frac_range, rr_ratio_range = strategy_risk_ranges(cfg)
    policy = resolve_trading_policy(cfg)
    use_mtf, mtf_windows = mtf_settings(cfg, arch)
    observation_schema_version = (
        "account-state-v3-repeated-payout"
        if policy.repeated_payout
        else str(cfg.env.get("observation_schema_version", "account-state-v1"))
    )
    eval_common = {
        "obs_window": cfg.env.obs_window,
        "initial_balance": cfg.account.initial_balance,
        "cost_model": cost_model(cfg),
        "guardrail_kwargs": eval_guardrail_kwargs(cfg),
        "risk_frac_range": risk_frac_range,
        "rr_ratio_range": rr_ratio_range,
        "sl_mode": str(cfg.risk.get("sl_mode", "fixed")),
        "tp_mode": str(cfg.risk.get("tp_mode", "fixed")),
        "tp_reference": str(cfg.risk.get("tp_reference", "structural")),
        "swing_buffer_pts": cfg.risk.swing_buffer_pts,
        "contract_size": cfg.account.contract_size,
        "max_loss_per_trade_usd": max_loss_per_trade(cfg),
        "dsr_eta": cfg.env.reward_dsr_eta,
        "continuous_actions": (args.algo == "sac") if args is not None else False,
        "use_sweep_reward": (args.reward == "sweep") if args is not None else False,
        "block_overnight": bool(cfg.env.get("block_overnight", True)),
        "eod_risk": dict(cfg.env.get("eod_risk", {})),
        "use_vae": getattr(args, "use_vae", False) if args is not None else False,
        "vae": env_vae,
        "strategy": strategy,
        "strategy_actions": strategy_actions,
        "entry_state_machine": bool(cfg.env.get("entry_state_machine", False)),
        "candidate_max_age_bars": int(cfg.env.get("candidate_max_age_bars", 5)),
        "arm_requires_retest": bool(cfg.env.get("arm_requires_retest", False)),
        "entry_state_observation": bool(cfg.env.get("entry_state_observation", True)),
        "strict_research_eligibility": bool(cfg.env.get("strict_research_eligibility", False)),
        "strategy_reward": strategy_reward,
        "strategy_weight": strategy_weight,
        "sl_buffer_pts": sl_buffer_pts,
        "pre_ny_by_date": pre_ny_by_date,
        "min_sl_atr_mult": float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        "min_sl_points": float(cfg.risk.get("min_sl_points", 0.0)),
        "max_entries_per_session": int(cfg.env.get("max_entries_per_session", 0)),
        "open_manipulation_bars": int(cfg.env.get("open_manipulation_bars", 10)),
        "entry_cooldown_bars": int(cfg.env.get("entry_cooldown_bars", 0)),
        "reward_mode": str(cfg.env.get("reward_mode", "dsr")),
        "fixed_risk_usd": float(cfg.env.get("fixed_risk_usd", 0.0)),
        "risk_mode": str(cfg.env.get("risk_mode", "dynamic")),
        "observation_schema_version": observation_schema_version,
        "observation_age_scale_bars": float(cfg.env.get("observation_age_scale_bars", 60.0)),
        "entry_intensity_threshold": float(cfg.env.get("entry_intensity_threshold", 0.0)),
        "peak_trailing_dd_limit": float(cfg.env.get("peak_trailing_dd_limit", 0.0)),
        "agent_direction_control": bool(cfg.env.get("agent_direction_control", False)),
        "direction_override_threshold": float(cfg.env.get("direction_override_threshold", 0.0)),
        "fill_delay_ms": fill_delay_ms(cfg),
        "randomized_fill_latency_ms": randomized_fill_latency_ms(cfg),
        "tick_fill_slippage_points": float(tick_fill_slippage_points),
        "tick_execution_required": tick_execution_required(cfg),
        "max_tick_quote_wait_ms": max_tick_quote_wait_ms(cfg),
        "execution_seed": (
            int(getattr(args, "seed"))
            if args is not None and getattr(args, "seed", None) is not None
            else None
        ),
        "allow_multi_tp": bool(cfg.env.get("allow_multi_tp", False)),
        "allow_simplex": bool(cfg.env.get("allow_simplex", True)),
        "allow_ema_exit": bool(cfg.env.get("allow_ema_exit", True)),
        "allow_agent_tp_mode": bool(cfg.env.get("allow_agent_tp_mode", False)),
        "allow_agent_sl_mode": bool(cfg.env.get("allow_agent_sl_mode", True)),
        "sl_mode_defer_threshold": float(cfg.env.get("sl_mode_defer_threshold", 0.1)),
        "tp_mode_defer_threshold": float(cfg.env.get("tp_mode_defer_threshold", 0.1)),
        "tp_breakeven_alpha": float(cfg.env.get("tp_breakeven_alpha", 0.5)),
        "risk_floor": float(cfg.env.get("risk_floor", 0.0005)),
        "mtf": use_mtf,
        "mtf_windows": mtf_windows,
        "trading_policy": policy.to_dict(),
    }
    return eval_common, strategy, strategy_reward, strategy_weight, strategy_actions


def fill_delay_ms(cfg: Any) -> int:
    """Milliseconds between the crossing tick and the SL/TP fill."""
    execution = cfg.get("execution") if hasattr(cfg, "get") else None
    if execution is None:
        return 0
    try:
        return int(execution.get("fill_delay_ms", 0))
    except Exception:
        return 0


def randomized_fill_latency_ms(cfg: Any) -> float:
    """Maximum exclusive randomized tick latency; zero means immediate tick."""
    execution = cfg.get("execution") if hasattr(cfg, "get") else None
    if execution is None:
        return 0.0
    try:
        return float(execution.get("randomized_fill_latency_ms", 0.0))
    except Exception:
        return 0.0


def tick_execution_required(cfg: Any) -> bool:
    """Whether the effective config forbids candle-priced execution fills."""
    costs = cfg.get("costs", {}) if hasattr(cfg, "get") else {}
    return bool(costs.get("use_tick_execution", False)) if hasattr(costs, "get") else False


def max_tick_quote_wait_ms(cfg: Any) -> int:
    """Maximum permitted delay from sampled arrival to the next tick quote."""
    execution = cfg.get("execution") if hasattr(cfg, "get") else None
    if execution is None:
        return 60_000
    try:
        return int(execution.get("max_tick_quote_wait_ms", 60_000))
    except Exception:
        return 60_000


def mtf_settings(cfg: Any, arch: str | None) -> tuple[bool, dict[str, int]]:
    """``mtf`` is on only when the selected architecture is the multi-window encoder."""
    from simulation.environment.observations import DEFAULT_HTF_WINDOWS

    chosen = arch
    if chosen is None:
        agent = cfg.get("agent") if hasattr(cfg, "get") else None
        chosen = str(agent.get("arch", "tcn")) if agent is not None else "tcn"
    windows = dict(DEFAULT_HTF_WINDOWS)
    encoder = cfg.get("encoder") if hasattr(cfg, "get") else None
    raw = encoder.get("windows") if encoder is not None and hasattr(encoder, "get") else None
    if raw:
        for key, value in dict(raw).items():
            windows[str(key)] = int(value)
    return str(chosen) == "mtf", windows


def cost_model(cfg: Any) -> CostModel:
    """Build the broker ``CostModel`` from ``costs.*``.

    ``make_env`` used to leave the env on its ``COST_US100`` default, so the
    ``costs.slippage_points: 5`` YAML value never reached fills. Spreads come
    from the primary-symbol YAML key; slippage is quoted in price points.
    """
    costs = cfg.get("costs", {}) if hasattr(cfg, "get") else {}
    if costs is None:
        costs = {}
    get = costs.get if hasattr(costs, "get") else lambda _k, _d=None: _d
    primary = str(cfg.data.primary) if hasattr(cfg, "data") else "US100.cash"
    spread_key = "spread_us500" if "US500" in primary else "spread_us100"
    return CostModel(
        spread_points=float(get(spread_key, 0.6)),
        slippage_points=float(get("slippage_points", 0.0)),
        commission_per_lot=float(get("commission_per_lot", 0.0)),
        point_size=float(get("point_size", 0.01)),
    )


def make_env(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    cfg: Any,
    *,
    algo: str,
    reward: str,
    episodic: bool = True,
    use_vae: bool = False,
    vae: Any | None = None,
    pre_ny_by_date: dict[Any, Any] | None = None,
    obs_features_mmap: str | None = None,
    tickbook: TickBook | None = None,
    arch: str | None = None,
) -> TradingEnv:
    continuous_actions = algo == "sac"
    use_sweep_reward = reward == "sweep"
    use_mtf, mtf_windows = mtf_settings(cfg, arch)
    strategy, strategy_reward, strategy_weight = strategy_from_cfg(cfg)
    risk_frac_range, rr_ratio_range = strategy_risk_ranges(cfg)
    policy = resolve_trading_policy(cfg)
    observation_schema_version = (
        "account-state-v3-repeated-payout"
        if policy.repeated_payout
        else str(cfg.env.get("observation_schema_version", "account-state-v1"))
    )
    return TradingEnv(
        bars=bars,
        features=features,
        obs_window=cfg.env.obs_window,
        initial_balance=cfg.account.initial_balance,
        cost_model=cost_model(cfg),
        guardrail_kwargs=guardrail_kwargs(cfg),
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=cfg.risk.swing_buffer_pts,
        sl_mode=str(cfg.risk.get("sl_mode", "fixed")),
        tp_mode=str(cfg.risk.get("tp_mode", "fixed")),
        tp_reference=str(cfg.risk.get("tp_reference", "structural")),
        contract_size=cfg.account.contract_size,
        max_loss_per_trade_usd=max_loss_per_trade(cfg),
        dsr_eta=cfg.env.reward_dsr_eta,
        max_episode_steps=max_episode_steps(cfg),
        episodic=episodic,
        continuous_actions=continuous_actions,
        use_sweep_reward=use_sweep_reward,
        strategy=strategy,
        strategy_actions=bool(cfg.env.get("strategy_actions", False)),
        entry_state_machine=bool(cfg.env.get("entry_state_machine", False)),
        candidate_max_age_bars=int(cfg.env.get("candidate_max_age_bars", 5)),
        arm_requires_retest=bool(cfg.env.get("arm_requires_retest", False)),
        entry_state_observation=bool(cfg.env.get("entry_state_observation", True)),
        observation_schema_version=observation_schema_version,
        observation_age_scale_bars=float(cfg.env.get("observation_age_scale_bars", 60.0)),
        strict_research_eligibility=bool(cfg.env.get("strict_research_eligibility", False)),
        sl_buffer_pts=float(cfg.env.get("sl_buffer_pts", 0.0)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        block_overnight=bool(cfg.env.get("block_overnight", True)),
        eod_risk=dict(cfg.env.get("eod_risk", {})),
        min_sl_atr_mult=float(cfg.risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(cfg.risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(cfg.env.get("max_entries_per_session", 0)),
        open_manipulation_bars=int(cfg.env.get("open_manipulation_bars", 10)),
        entry_cooldown_bars=int(cfg.env.get("entry_cooldown_bars", 0)),
        reward_mode=str(cfg.env.get("reward_mode", "dsr")),
        fixed_risk_usd=float(cfg.env.get("fixed_risk_usd", 0.0)),
        risk_mode=str(cfg.env.get("risk_mode", "dynamic")),
        entry_intensity_threshold=float(cfg.env.get("entry_intensity_threshold", 0.0)),
        peak_trailing_dd_limit=float(cfg.env.get("peak_trailing_dd_limit", 0.0)),
        agent_direction_control=bool(cfg.env.get("agent_direction_control", False)),
        direction_override_threshold=float(cfg.env.get("direction_override_threshold", 0.0)),
        allow_agent_sl_mode=bool(cfg.env.get("allow_agent_sl_mode", True)),
        sl_mode_defer_threshold=float(cfg.env.get("sl_mode_defer_threshold", 0.1)),
        allow_agent_tp_mode=bool(cfg.env.get("allow_agent_tp_mode", False)),
        tp_mode_defer_threshold=float(cfg.env.get("tp_mode_defer_threshold", 0.1)),
        allow_multi_tp=bool(cfg.env.get("allow_multi_tp", False)),
        allow_simplex=bool(cfg.env.get("allow_simplex", True)),
        allow_ema_exit=bool(cfg.env.get("allow_ema_exit", True)),
        tp_breakeven_alpha=float(cfg.env.get("tp_breakeven_alpha", 0.5)),
        risk_floor=float(cfg.env.get("risk_floor", 0.0005)),
        tickbook=tickbook,
        tick_execution_required=tick_execution_required(cfg),
        fill_delay_ms=fill_delay_ms(cfg),
        randomized_fill_latency_ms=randomized_fill_latency_ms(cfg),
        max_tick_quote_wait_ms=max_tick_quote_wait_ms(cfg),
        tick_fill_slippage_points=0.0,
        use_vae=use_vae,
        vae=vae,
        pre_ny_by_date=pre_ny_by_date,
        obs_features_mmap=obs_features_mmap,
        mtf=use_mtf,
        mtf_windows=mtf_windows,
        trading_policy=policy.to_dict(),
    )
