from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading.execution.broker import Position
from trading.policy.legacy import uses_fixed_risk_budget
from trading.risk.budget import approve_dynamic_risk
from trading.risk.execution_reserve import modeled_execution_reserve_usd
from trading.risk.sizing import compute_lots

from .mixin_base import RuntimeMixinBase


class RiskRuntimeMixin(RuntimeMixinBase):
    position: Position | None

    def _check_guardrails(
        self,
        bar_time: Any,
        session_id: int,
        fill_bid: float,
        fill_ask: float,
        fill_time: Any | None = None,
    ) -> tuple[str | None, bool, bool, bool]:
        """Check guardrail breaches and apply forced closes.

        Returns ``(reason, session_blocked, done, truncated)`` where:

        - ``reason`` is the breach reason string or ``None`` if no breach.
        - ``session_blocked`` is ``True`` when the session is blocked after an
          earlier breach.
        - ``done`` / ``truncated`` follow the episode mode: in episodic mode
          a breach ends the episode; in non-episodic mode the split rollout
          continues with the breached session/year blocked as appropriate.

        When a breach occurs, any open position is force-closed and the
        breach is recorded in ``breach_log`` / ``breach_events``.
        """
        if self.repeated_payout_policy:
            reason = self.guardrails.breach_reason(self.account)
            session_blocked = False
        elif self.episodic:
            reason = self.guardrails.breach_reason(self.account)
            session_blocked = False
        elif self._year_failed:
            session_blocked = True
            reason = None
        else:
            session_blocked = session_id in self.breached_sessions
            reason = None if session_blocked else self.guardrails.breach_reason(self.account)
            if reason:
                self.breached_sessions.add(session_id)
                session_blocked = True
                if reason in ("trailing_dd", "max_drawdown"):
                    self._year_failed = True

        if reason:
            if self.position is not None:
                execution_meta: dict[str, Any] = {}
                if getattr(self, "tick_execution_required", False):
                    quote, execution_meta = self._resolve_randomized_tick_quote(
                        pd.Timestamp(fill_time if fill_time is not None else bar_time)
                    )
                    fill_bid, fill_ask = quote.bid, quote.ask
                    fill_time = quote.tick_time
                pnl, fill_price = self._close_at_quote(
                    self.position,
                    (fill_bid, fill_ask),
                    tick_fill=bool(execution_meta),
                )
                record = {
                    "type": "forced_close",
                    "pnl": pnl,
                    "price": fill_price,
                    "reason": reason,
                    "bar": self.step_idx,
                    "time": bar_time,
                    "decision_time": bar_time,
                    "fill_time": fill_time if fill_time is not None else bar_time,
                    "equity": self.account.equity,
                    **execution_meta,
                }
                self._stamp_exit(record, fill_price)
                self.trade_log.append(record)
                self.position = None
                self.sessions_with_trades.add(session_id)
                self._note_close(pnl)
            self.breach_log.append(reason)
            self.breach_events.append(
                {
                    "time": bar_time,
                    "session_id": session_id,
                    "reason": reason,
                    "equity": self.account.equity,
                    "fill_time": fill_time if fill_time is not None else bar_time,
                }
            )
            if self.repeated_payout_policy:
                self.cycle_failures += 1
                self.cycle_events.append(
                    {
                        "type": "cycle_failure",
                        "time": bar_time,
                        "cycle": int(self.cycle_count),
                        "outcome": "failure",
                        "reason": reason,
                        "equity": float(self.account.equity),
                        "cumulative_payouts": float(self.cumulative_payouts),
                        "total_economic_value": float(
                            self.account.equity + self.cumulative_payouts
                        ),
                    }
                )
            # Repeated-payout breaches fail the sustainability path even during
            # chronological evaluation. Legacy non-episodic evaluation retains
            # its session/year latch behavior.
            done = self.episodic or self.repeated_payout_policy
            truncated = False
        elif session_blocked:
            done = False
            truncated = False
        else:
            done = False
            truncated = False

        return reason, session_blocked, done, truncated

    def _apply_peak_trailing_dd(
        self, fill_bid: float, fill_ask: float, bar_time: Any, fill_time: Any | None = None
    ) -> bool:
        """Keep peak trailing drawdown inside the training ceiling.

        This is strategy behavior, not an FTMO kill and not a breakeven stop.
        At the ceiling, the open position is closed at the current quote.
        Its stop and target prices are left unchanged. New entries are refused
        while drawdown from the peak is still at the ceiling.
        """
        limit = self.peak_trailing_dd_limit
        if limit <= 0.0:
            return False
        if float(self.account.trailing_drawdown_pct()) < limit:
            return False
        self._peak_dd_fired = True
        if self.position is not None:
            execution_meta: dict[str, Any] = {}
            if getattr(self, "tick_execution_required", False):
                order_time = (
                    pd.Timestamp(fill_time)
                    if fill_time is not None
                    else pd.Timestamp(bar_time) + pd.Timedelta(minutes=self._minutes_per_step)
                )
                quote, execution_meta = self._resolve_randomized_tick_quote(order_time)
                fill_bid, fill_ask = quote.bid, quote.ask
                fill_time = quote.tick_time
            pnl, fill_price = self._close_at_quote(
                self.position,
                (fill_bid, fill_ask),
                tick_fill=bool(execution_meta),
            )
            record = {
                "type": "close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "peak_dd_behavior",
                "bar": self.step_idx,
                "time": bar_time,
                "decision_time": bar_time,
                "fill_time": fill_time if fill_time is not None else bar_time,
                "equity": self.account.equity,
                **execution_meta,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(self._current_session_id())
            self._note_close(pnl)
        return True

    def _sizing_equity(self) -> float:
        """Equity basis for lot sizing (dynamic risk mode only).

        In ``risk_mode="dynamic"`` the agent's ``risk_frac`` acts on live account
        equity, so risk per trade drifts with the balance. In ``risk_mode="fixed"``
        sizing never touches this value: the dollar budget comes from
        ``fixed_risk_usd`` via :meth:`_risk_budget_usd` and is passed to
        ``compute_lots`` as an explicit amount, which is what makes 1R exactly
        ``fixed_risk_usd`` at every stop width.
        """
        return float(self.account.equity)

    def _risk_budget_usd(self, risk_frac: float) -> float:
        """Dollar risk actually requested for the next trade, by sizing mode.

        ``fixed``  -> ``fixed_risk_usd``, independent of equity and risk_frac.
        ``dynamic`` -> live equity times the agent's chosen fraction.
        """
        if uses_fixed_risk_budget(
            self.risk_mode, bool(getattr(self, "repeated_payout_policy", False))
        ):
            return float(self.fixed_risk_usd)
        requested = max(0.0, float(self.account.equity) * float(risk_frac))
        self._last_requested_risk_usd = requested
        if not bool(getattr(self, "repeated_payout_policy", False)):
            self._last_approved_risk_usd = requested
            self._last_execution_reserve_usd = 0.0
            return requested
        decision = approve_dynamic_risk(
            equity=self.account.equity,
            risk_frac=risk_frac,
            max_agent_risk_frac=self.trading_policy.max_agent_risk_frac,
            daily_capacity=self.account.remaining_daily_capacity(self.guardrails.daily_loss_limit),
            account_capacity=self.account.remaining_account_capacity(
                self.trading_policy.account_floor
            ),
            execution_reserve_usd=self.trading_policy.execution_reserve_usd,
        )
        self._last_requested_risk_usd = decision.requested_usd
        self._last_approved_risk_usd = decision.approved_usd
        self._last_execution_reserve_usd = decision.execution_reserve_usd
        return decision.approved_usd

    def _modeled_execution_reserve_usd(self, lots: float) -> float:
        """Variable execution loss allowance implied by the active cost model.

        Entry spread is already embedded in the actual bid/ask ``entry_price``
        used by stop-risk sizing. This reserve therefore covers adverse stop
        slippage plus explicit commission on both entry and exit.
        """
        return modeled_execution_reserve_usd(
            lots=lots,
            slippage_points=self.cost_model.slippage_points,
            commission_per_lot=self.cost_model.commission_per_lot,
            contract_size=self.contract_size,
        )

    def _compute_lots_for(self, risk_frac: float, entry_price: float, sl_price: float) -> float:
        """Lot size for one trade under the active sizing mode.

        Both sizing modes honor the resolved per-trade policy and the configured
        $900 sizing ceiling. The percentage limit is recalculated from current
        equity for each decision; no minimum trade-risk threshold is imposed.
        """
        if bool(getattr(self, "repeated_payout_policy", False)):
            approved_total_risk = self._risk_budget_usd(risk_frac)
            lots = self._lots_with_execution_reserve(
                approved_total_risk,
                risk_frac,
                entry_price,
                sl_price,
            )
            self._last_execution_reserve_usd = float(
                self.trading_policy.execution_reserve_usd
            ) + self._modeled_execution_reserve_usd(lots)
            return lots
        requested_total_risk = self._risk_budget_usd(risk_frac)
        per_trade_limit = min(
            requested_total_risk,
            max(0.0, float(self.max_loss_per_trade_usd)),
            self.guardrails.risk_per_trade_limit_for(self.account),
        )
        approved_total_risk = per_trade_limit
        self._last_approved_risk_usd = approved_total_risk
        if self.risk_mode == "fixed":
            return compute_lots(
                self._sizing_equity(),
                risk_frac,
                entry_price,
                sl_price,
                contract_size=self.contract_size,
                min_lot=self.min_lot,
                max_lot=self.max_lot,
                max_loss_cap=None,
                risk_usd=approved_total_risk,
            )
        lots = self._lots_with_execution_reserve(
            approved_total_risk,
            risk_frac,
            entry_price,
            sl_price,
        )
        self._last_execution_reserve_usd = self._modeled_execution_reserve_usd(lots)
        return lots

    def _lots_with_execution_reserve(
        self,
        total_risk_budget_usd: float,
        risk_frac: float,
        entry_price: float,
        sl_price: float,
    ) -> float:
        """Size stop exposure so stop loss plus modeled costs fit the budget."""
        stop_risk_per_lot = abs(float(entry_price) - float(sl_price)) * self.contract_size
        if total_risk_budget_usd <= 0.0 or stop_risk_per_lot <= 0.0:
            return 0.0
        execution_reserve_per_lot = self._modeled_execution_reserve_usd(1.0)
        all_in_risk_per_lot = stop_risk_per_lot + execution_reserve_per_lot
        if all_in_risk_per_lot <= 0.0:
            return 0.0
        stop_risk_budget = total_risk_budget_usd * stop_risk_per_lot / all_in_risk_per_lot
        lots = compute_lots(
            self._sizing_equity(),
            risk_frac,
            entry_price,
            sl_price,
            contract_size=self.contract_size,
            min_lot=self.min_lot,
            max_lot=self.max_lot,
            max_loss_cap=None,
            risk_usd=stop_risk_budget,
        )
        lots_at_budget = total_risk_budget_usd / all_in_risk_per_lot
        if lots_at_budget < self.min_lot:
            return 0.0
        return float(min(lots, lots_at_budget))

    def _actual_risk_usd(self, stop_distance: float) -> float:
        """Dollar risk the filled position really carries, in account currency.

        Uses the broker's accepted size rather than the requested budget, so the
        logged number is what a stop would cost. Returns NaN when there is no
        open position or the stop distance is unusable.
        """
        pos = self.position
        if pos is None or not np.isfinite(stop_distance) or stop_distance <= 0.0:
            return float("nan")
        return float(pos.size) * float(stop_distance) * float(self.contract_size)

    def _eod_adverse_cut_usd(self, pos: Position) -> float:
        """Dollar adverse excursion that fires the intraday max-loss guard.

        The guard must never sit *inside* the trade's own structural stop, or
        a position sized to risk ``ftmo.risk_per_trade_limit`` gets flattened
        at half that risk and never reaches its target. When
        ``respect_planned_risk`` is on, the cutoff is the larger of the
        configured cap and ``planned_risk_mult`` times the loss the position
        would take at its structural stop.
        """
        configured = float(self._eod_max_loss_usd)
        if not self._eod_respect_planned_risk:
            return configured
        planned = float(pos.planned_risk_usd or 0.0)
        if planned <= 0.0:
            return configured
        return float(max(configured, float(self._eod_planned_risk_mult) * planned))
