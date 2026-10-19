"""Selectable trading-policy configuration shared by train, workers, and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


def _float_or_default(value: Any, default: float) -> float:
    return default if value is None else float(value)


@dataclass(frozen=True)
class TradingPolicyConfig:
    mode: str = "legacy"
    operating_balance: float = 100_000.0
    profit_target_balance: float = 110_000.0
    account_floor: float = 90_000.0
    daily_loss_limit: float = 5_000.0
    min_agent_risk_frac: float = 0.0
    max_agent_risk_frac: float = 0.009
    execution_reserve_usd: float = 0.0
    payout_trigger: str = "realized_balance_flat"
    continue_after_payout: bool = True
    trailing_dd_limit: float = 0.0
    enforce_remaining_daily_capacity: bool = True
    enforce_remaining_account_capacity: bool = True
    repeated_eval: bool = True

    @property
    def repeated_payout(self) -> bool:
        return self.mode == "repeated_payout"

    def validate(self) -> TradingPolicyConfig:
        if self.mode not in {"legacy", "repeated_payout"}:
            raise ValueError(f"unknown trading_policy.mode: {self.mode!r}")
        if self.repeated_payout:
            if self.profit_target_balance <= self.operating_balance:
                raise ValueError("repeated payout target must exceed operating balance")
            if self.account_floor >= self.operating_balance:
                raise ValueError("repeated payout account floor must be below operating balance")
            if self.daily_loss_limit < 0.0:
                raise ValueError("repeated payout daily loss limit cannot be negative")
            if self.min_agent_risk_frac != 0.0:
                raise ValueError("repeated payout minimum agent risk must be zero")
            if not 0.0 <= self.max_agent_risk_frac < 0.01:
                raise ValueError("repeated payout max agent risk must be in [0, 0.01)")
            if self.execution_reserve_usd < 0.0:
                raise ValueError("execution reserve cannot be negative")
            if self.payout_trigger != "realized_balance_flat":
                raise ValueError("repeated payout trigger must be realized_balance_flat")
            if not self.continue_after_payout:
                raise ValueError("repeated payout evaluation must continue after payout")
            if self.trailing_dd_limit != 0.0:
                raise ValueError("repeated payout account trailing drawdown must be disabled")
            if not self.enforce_remaining_daily_capacity:
                raise ValueError("repeated payout must enforce remaining daily capacity")
            if not self.enforce_remaining_account_capacity:
                raise ValueError("repeated payout must enforce remaining account capacity")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_trading_policy(cfg: Any) -> TradingPolicyConfig:
    block = cfg.get("trading_policy", {}) if hasattr(cfg, "get") else {}
    if block is None:
        block = {}
    mode = str(block.get("mode", "legacy")).strip().lower()
    repeated = block.get("repeated_payout", {}) or {}
    account = cfg.get("account", {}) if hasattr(cfg, "get") else {}
    initial = float(account.get("initial_balance", 100_000.0))
    result = TradingPolicyConfig(
        mode=mode,
        operating_balance=float(repeated.get("operating_balance", initial)),
        profit_target_balance=float(
            repeated.get("payout_target_balance", repeated.get("profit_target_balance", 110_000.0))
        ),
        account_floor=float(
            repeated.get("account_floor_equity", repeated.get("account_floor", 90_000.0))
        ),
        daily_loss_limit=float(repeated.get("daily_loss_limit", 5_000.0)),
        min_agent_risk_frac=_float_or_default(
            repeated.get("min_agent_risk_frac")
            if repeated.get("min_agent_risk_frac") is not None
            else repeated.get("agent_risk_min_frac"),
            0.0,
        ),
        max_agent_risk_frac=_float_or_default(
            repeated.get("max_agent_risk_frac")
            if repeated.get("max_agent_risk_frac") is not None
            else repeated.get("agent_risk_max_frac"),
            0.009,
        ),
        execution_reserve_usd=_float_or_default(
            repeated.get("execution_reserve_usd")
            if repeated.get("execution_reserve_usd") is not None
            else repeated.get("reserve_cost_buffer_usd"),
            0.0,
        ),
        payout_trigger=str(repeated.get("payout_trigger", "realized_balance_flat")),
        continue_after_payout=bool(repeated.get("continue_after_payout", True)),
        trailing_dd_limit=float(repeated.get("trailing_dd_limit", 0.0)),
        enforce_remaining_daily_capacity=bool(
            repeated.get("enforce_remaining_daily_capacity", True)
        ),
        enforce_remaining_account_capacity=bool(
            repeated.get("enforce_remaining_account_capacity", True)
        ),
        repeated_eval=bool(repeated.get("repeated_eval", True)),
    )
    return result.validate()


def resolve_trading_policy_mapping(cfg: Mapping[str, Any]) -> TradingPolicyConfig:
    return resolve_trading_policy(dict(cfg))
