"""Lightweight constructor for a bundle-backed trading environment."""

from __future__ import annotations

from typing import Any

from trading.execution.costs import CostModel
from trading.policy.config import resolve_trading_policy_mapping
from trading.strategies import BaselineStrategy, DistributionStrategy, PO3IFVGStrategy

from .distribution_reward import DistributionReward
from .env import TradingEnv
from .po3_reward import PO3Reward
from .spec import EnvSpec


def _block(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _strategy(config: dict[str, Any], strategy_actions: bool) -> tuple[Any, Any, float]:
    """Recreate the configured strategy and its optional shaping reward."""
    if not strategy_actions:
        return BaselineStrategy(), None, 0.0
    values = _block(config, "strategy")
    reward_cfg = values.get("reward", {}) or {}
    entry_cfg = values.get("entry", {}) or {}
    name = str(values.get("name", "baseline"))
    gate = bool(entry_cfg.get("enforce_gate", False))
    manipulation = str(values.get("manipulation_filter", "against"))
    weight = float(reward_cfg.get("strategy_weight", 0.0))
    strategy: Any
    reward: Any
    if name == "po3_ifvg":
        ifvg = values.get("ifvg", {}) or {}
        env_values = _block(config, "env")
        strategy = PO3IFVGStrategy(
            enforce_gate=gate,
            require_price_retest=bool(ifvg.get("require_price_retest", True)),
            entry_mode=str(entry_cfg.get("mode", "legacy")),
            candidate_max_age_bars=int(env_values.get("candidate_max_age_bars", 5)),
            variant_id=str(values.get("variant_id", "po3_ifvg_legacy")),
        )
        strategy.manipulation_filter = manipulation
        reward = PO3Reward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            manipulation_penalty=float(reward_cfg.get("manipulation_penalty", 0.02)),
            invalid_ifvg_penalty=float(reward_cfg.get("invalid_ifvg_penalty", 0.01)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
        )
        return strategy, reward, weight
    if name == "distribution":
        features = _block(config, "features")
        pd_gate = bool(entry_cfg.get("previous_session_pd_gate", False))
        if pd_gate and not bool(features.get("include_previous_session_pd", False)):
            raise ValueError(
                "strategy.entry.previous_session_pd_gate requires "
                "features.include_previous_session_pd=true"
            )
        strategy = DistributionStrategy(
            enforce_gate=gate,
            entry_mode=str(entry_cfg.get("mode", "legacy")),
            candidate_max_age_bars=int(_block(config, "env").get("candidate_max_age_bars", 5)),
            previous_session_pd_gate=pd_gate,
            variant_id=str(values.get("variant_id", "distribution_legacy")),
        )
        strategy.manipulation_filter = manipulation
        reward = DistributionReward(
            entry_bonus=float(reward_cfg.get("entry_bonus", 0.01)),
            sweep_penalty=float(reward_cfg.get("sweep_penalty", 0.02)),
            distribution_bonus=float(reward_cfg.get("distribution_bonus", 0.005)),
        )
        return strategy, reward, weight
    return BaselineStrategy(), None, 0.0


def cost_model(config: dict[str, Any]) -> CostModel:
    costs = _block(config, "costs")
    primary = str(_block(config, "data").get("primary", "US100.cash"))
    spread_key = "spread_us500" if "US500" in primary else "spread_us100"
    return CostModel(
        spread_points=float(costs.get(spread_key, 0.6)),
        slippage_points=float(costs.get("slippage_points", 0.0)),
        commission_per_lot=float(costs.get("commission_per_lot", 0.0)),
        point_size=float(costs.get("point_size", 0.01)),
    )


def max_episode_steps(config: dict[str, Any]) -> int | None:
    value = _block(config, "env").get("max_episode_steps")
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"}):
        return None
    return int(value)


def make_bundle_env(spec: EnvSpec) -> TradingEnv:
    """Open the bundle and reconstruct one env using only small config values."""
    if spec.use_vae:
        raise ValueError(
            "bundle workers with VAE require the separately approved latent-table phase"
        )
    cfg = dict(spec.config)
    env = _block(cfg, "env")
    risk = _block(cfg, "risk")
    account = _block(cfg, "account")
    strategy_actions = bool(env.get("strategy_actions", False))
    policy = resolve_trading_policy_mapping(cfg)
    ftmo = _block(cfg, "ftmo")
    strategy, strategy_reward, strategy_weight = _strategy(cfg, strategy_actions)
    if policy.repeated_payout:
        strategy_risk = _block(cfg, "strategy").get("risk", {}) or {}
        risk_frac_range = (0.0, policy.max_agent_risk_frac)
        rr_ratio_range = tuple(float(x) for x in strategy_risk.get("rr_range", [1.5, 5.0]))
        max_loss = policy.operating_balance * policy.max_agent_risk_frac
    elif strategy_actions:
        strategy_risk = _block(cfg, "strategy").get("risk", {}) or {}
        risk_frac_values = tuple(
            float(x) for x in strategy_risk.get("risk_frac_range", [0.005, 0.01])
        )
        if len(risk_frac_values) != 2:
            raise ValueError("strategy.risk.risk_frac_range must contain two values")
        risk_frac_range = (risk_frac_values[0], risk_frac_values[1])
        rr_ratio_range = tuple(float(x) for x in strategy_risk.get("rr_range", [1.5, 5.0]))
        hard_limit = float(ftmo.get("risk_per_trade_limit", 1000.0))
        sizing_limit = float(ftmo.get("risk_sizing_limit_usd", hard_limit))
        max_loss = min(hard_limit, sizing_limit)
    else:
        default_risk = float(risk.get("default_risk_frac", 0.01))
        default_rr = float(risk.get("rr_ratio_default", 2.0))
        risk_frac_range = (default_risk * 0.5, default_risk * 2.0)
        rr_ratio_range = (default_rr * 0.5, default_rr * 1.5)
        hard_limit = float(ftmo.get("risk_per_trade_limit", 1000.0))
        sizing_limit = float(ftmo.get("risk_sizing_limit_usd", hard_limit))
        max_loss = min(hard_limit, sizing_limit)
    guardrail_kwargs: dict[str, Any] = {
        "limit_value_mode": str(ftmo.get("limit_value_mode", "fixed_usd")),
        "daily_loss_limit": float(ftmo.get("daily_loss_limit", 5000.0)),
        "max_loss_limit": float(ftmo.get("max_loss_limit", 10000.0)),
        "risk_per_trade_limit": float(ftmo.get("risk_per_trade_limit", 1000.0)),
        "soft_daily_loss_limit": float(ftmo.get("soft_daily_loss_limit", 5000.0)),
        "soft_max_loss_limit": float(ftmo.get("soft_max_loss_limit", 10000.0)),
        "trailing_dd_limit": float(ftmo.get("trailing_dd_limit", 0.0)),
        "soft_trailing_dd_limit": float(ftmo.get("soft_trailing_dd_limit", 0.0)),
        "risk_per_trade_limit_pct": float(ftmo.get("risk_per_trade_limit_pct", 0.01)),
        "daily_loss_limit_pct": float(ftmo.get("daily_loss_limit_pct", 0.05)),
        "soft_daily_loss_limit_pct": float(ftmo.get("soft_daily_loss_limit_pct", 0.05)),
        "max_loss_limit_initial_pct": float(ftmo.get("max_loss_limit_initial_pct", 0.10)),
        "soft_max_loss_limit_initial_pct": float(ftmo.get("soft_max_loss_limit_initial_pct", 0.10)),
    }
    if policy.repeated_payout:
        guardrail_kwargs.update(
            daily_loss_limit=policy.daily_loss_limit,
            max_loss_limit=policy.operating_balance - policy.account_floor,
            risk_per_trade_limit=policy.operating_balance * policy.max_agent_risk_frac,
            soft_daily_loss_limit=0.0,
            soft_max_loss_limit=0.0,
            trailing_dd_limit=0.0,
            soft_trailing_dd_limit=0.0,
            limit_value_mode="fixed_usd",
            risk_per_trade_limit_pct=0.0,
            daily_loss_limit_pct=0.0,
            soft_daily_loss_limit_pct=0.0,
            max_loss_limit_initial_pct=0.0,
            soft_max_loss_limit_initial_pct=0.0,
        )
    if spec.evaluation:
        # Match train_rl.eval_guardrail_kwargs: evaluation reports apply the
        # daily and initial-equity loss gates, while trailing drawdown remains
        # a training kill switch only.
        guardrail_kwargs["trailing_dd_limit"] = 0.0
        guardrail_kwargs["soft_trailing_dd_limit"] = 0.0
    encoder = _block(cfg, "encoder")
    windows = {str(key): int(value) for key, value in (encoder.get("windows", {}) or {}).items()}
    costs = _block(cfg, "costs")
    execution = _block(cfg, "execution")
    return TradingEnv.from_bundle(
        spec.bundle_dir,
        obs_window=int(env.get("obs_window", 60)),
        initial_balance=float(account.get("initial_balance", 100_000.0)),
        cost_model=cost_model(cfg),
        guardrail_kwargs=guardrail_kwargs,
        risk_frac_range=risk_frac_range,
        rr_ratio_range=rr_ratio_range,
        swing_buffer_pts=float(risk.get("swing_buffer_pts", 1.0)),
        sl_mode=str(risk.get("sl_mode", "fixed")),
        contract_size=float(account.get("contract_size", 1.0)),
        max_loss_per_trade_usd=max_loss,
        dsr_eta=float(env.get("reward_dsr_eta", 0.01)),
        max_episode_steps=max_episode_steps(cfg),
        episodic=spec.episodic,
        continuous_actions=spec.algo == "sac",
        use_sweep_reward=spec.reward == "sweep",
        strategy=strategy,
        strategy_actions=strategy_actions,
        entry_state_machine=bool(env.get("entry_state_machine", False)),
        candidate_max_age_bars=int(env.get("candidate_max_age_bars", 5)),
        arm_requires_retest=bool(env.get("arm_requires_retest", False)),
        entry_state_observation=bool(env.get("entry_state_observation", True)),
        observation_schema_version=(
            "account-state-v3-repeated-payout"
            if policy.repeated_payout
            else str(env.get("observation_schema_version", "account-state-v1"))
        ),
        observation_age_scale_bars=float(env.get("observation_age_scale_bars", 60.0)),
        strict_research_eligibility=bool(env.get("strict_research_eligibility", False)),
        sl_buffer_pts=float(env.get("sl_buffer_pts", 0.0)),
        strategy_reward=strategy_reward,
        strategy_weight=strategy_weight,
        block_overnight=bool(env.get("block_overnight", True)),
        eod_risk=dict(env.get("eod_risk", {}) or {}),
        min_sl_atr_mult=float(risk.get("min_sl_atr_mult", 0.5)),
        min_sl_points=float(risk.get("min_sl_points", 0.0)),
        max_entries_per_session=int(env.get("max_entries_per_session", 0)),
        open_manipulation_bars=int(env.get("open_manipulation_bars", 10)),
        entry_cooldown_bars=int(env.get("entry_cooldown_bars", 0)),
        reward_mode=str(env.get("reward_mode", "dsr")),
        fixed_risk_usd=float(env.get("fixed_risk_usd", 0.0)),
        risk_mode=str(env.get("risk_mode", "dynamic")),
        entry_intensity_threshold=float(env.get("entry_intensity_threshold", 0.0)),
        peak_trailing_dd_limit=float(env.get("peak_trailing_dd_limit", 0.0)),
        agent_direction_control=bool(env.get("agent_direction_control", False)),
        direction_override_threshold=float(env.get("direction_override_threshold", 0.0)),
        allow_agent_sl_mode=bool(env.get("allow_agent_sl_mode", True)),
        sl_mode_defer_threshold=float(env.get("sl_mode_defer_threshold", 0.1)),
        allow_agent_tp_mode=bool(env.get("allow_agent_tp_mode", False)),
        tp_mode_defer_threshold=float(env.get("tp_mode_defer_threshold", 0.1)),
        allow_multi_tp=bool(env.get("allow_multi_tp", False)),
        allow_simplex=bool(env.get("allow_simplex", True)),
        tp_breakeven_alpha=float(env.get("tp_breakeven_alpha", 0.5)),
        risk_floor=float(env.get("risk_floor", 0.0005)),
        tick_execution_required=bool(costs.get("use_tick_execution", False)),
        fill_delay_ms=int(execution.get("fill_delay_ms", 0)),
        randomized_fill_latency_ms=float(execution.get("randomized_fill_latency_ms", 0.0)),
        max_tick_quote_wait_ms=int(execution.get("max_tick_quote_wait_ms", 60_000)),
        mtf=spec.arch == "mtf",
        mtf_windows=windows,
        trading_policy=policy.to_dict(),
    )
