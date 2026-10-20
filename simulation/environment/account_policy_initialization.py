"""Focused simulation initialization service."""

from __future__ import annotations

from typing import Any

from trading.execution.broker import Broker
from trading.execution.costs import CostModel
from trading.policy.config import TradingPolicyConfig
from trading.risk.guardrails import FTMOGuardrails

from .mixin_base import RuntimeMixinBase
from .position_constants import _SL_MODE_RANK, _TP_MODE_RANK


class AccountPolicyInitializationMixin(RuntimeMixinBase):
    def _initialize_account_policy(
        self,
        obs_window: int,
        initial_balance: float,
        cost_model: CostModel,
        broker_kwargs: dict[str, Any] | None,
        trading_policy: dict[str, Any] | None,
        guardrail_kwargs: dict[str, Any] | None,
        risk_frac_range: tuple[float, float],
        rr_ratio_range: tuple[float, float],
        sl_buffer_pts: float,
        sl_mode: str,
        tp_mode: str,
        tp_reference: str,
        min_lot: float,
        max_lot: float,
        contract_size: float,
        max_loss_per_trade_usd: float,
        block_overnight: bool,
        eod_risk: dict[str, Any] | None,
    ) -> None:
        self.obs_window = obs_window
        self.initial_balance = initial_balance
        self.cost_model = cost_model
        self.broker = Broker(cost_model=cost_model, **(broker_kwargs or {}))
        policy_values = dict(trading_policy or {})
        self.trading_policy = TradingPolicyConfig(**policy_values).validate()
        self.repeated_payout_policy = self.trading_policy.repeated_payout
        if (
            self.repeated_payout_policy
            and abs(float(initial_balance) - self.trading_policy.operating_balance) > 1e-9
        ):
            raise ValueError("repeated-payout initial balance must equal operating_balance")
        resolved_guardrails = dict(guardrail_kwargs or {})
        if self.repeated_payout_policy:
            resolved_guardrails.update(
                daily_loss_limit=self.trading_policy.daily_loss_limit,
                max_loss_limit=self.trading_policy.operating_balance
                - self.trading_policy.account_floor,
                risk_per_trade_limit=self.trading_policy.operating_balance
                * self.trading_policy.max_agent_risk_frac,
                trailing_dd_limit=0.0,
                soft_trailing_dd_limit=0.0,
            )
        self.guardrails = FTMOGuardrails(**resolved_guardrails)

        self.risk_frac_range = (
            (0.0, self.trading_policy.max_agent_risk_frac)
            if self.repeated_payout_policy
            else risk_frac_range
        )
        self.rr_ratio_range = rr_ratio_range
        self.swing_buffer_pts = float(sl_buffer_pts)
        self.sl_mode = str(sl_mode or "fixed").lower()
        if self.sl_mode not in _SL_MODE_RANK:
            raise ValueError(f"Unknown sl_mode: {self.sl_mode!r}")
        self.tp_mode = str(tp_mode or "fixed").lower()
        if self.tp_mode not in _TP_MODE_RANK:
            raise ValueError(f"Unknown tp_mode: {self.tp_mode!r}")
        self.tp_reference = str(tp_reference or "structural")
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.contract_size = contract_size
        self.max_loss_per_trade_usd = max_loss_per_trade_usd
        self.block_overnight = bool(block_overnight)
        eod = dict(eod_risk or {})
        self._eod_max_loss_usd = float(eod.get("max_loss_usd", 500.0))
        self._eod_max_age_hours = float(eod.get("max_age_hours", 8.0))
        self._eod_stale_bars = int(eod.get("stale_bars", 60))
        self._eod_progress_atr = float(eod.get("progress_threshold_atr", 0.25))
        # Risk/sizing alignment: the intraday adverse-excursion cut never
        # fires closer than the trade's own planned loss at the structural
        # stop. Without this, a $500 cutoff flattens a position sized to
        # risk $1,000 and the structural thesis never gets to play out.
        self._eod_respect_planned_risk = bool(eod.get("respect_planned_risk", True))
        self._eod_planned_risk_mult = float(eod.get("planned_risk_mult", 1.1))
        # Breakeven management: once favorable excursion reaches this many
        # stop distances, the stop moves to the entry plus a small buffer.
        # 0 disables the rewrite.
        self._breakeven_trigger_r = float(eod.get("breakeven_trigger_r", 0.0))
        self._breakeven_buffer_pts = float(eod.get("breakeven_buffer_pts", 2.0))


__all__ = ["AccountPolicyInitializationMixin"]
