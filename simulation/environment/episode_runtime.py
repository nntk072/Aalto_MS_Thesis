from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_data.loading.ticks import TickExecutionUnavailable, resolve_tick_execution_by_deadline
from market_data.transforms.sessions import ny_session_mask
from simulation.contracts import SimulationTransition
from trading.execution.broker import Position
from trading.policy.repeated_payout import evaluate_payout

from .market_view import BarView, FeatureRow
from .mixin_base import RuntimeMixinBase
from .sweep_reward import CompositeReward


class EpisodeRuntimeMixin(RuntimeMixinBase):
    prev_session: int | None
    position: Position | None
    _last_realized_close_pnl: float | None
    _last_realized_close_r: float | None
    _pending_realized_r: float | None

    def _step_untraced(
        self,
        action: int | float | np.ndarray[Any, Any],
    ) -> SimulationTransition[dict[str, np.ndarray[Any, Any]]]:
        """Execute one environment step through explicit semantic stages."""
        if self.step_idx >= self._n_bars or self._ny_pos >= len(self._ny_indices):
            return SimulationTransition(self._get_observation(), 0.0, False, True, {})

        bar = self._bar_at(self.step_idx)
        feat_row = self._feature_row_at(self.step_idx)
        bar_time = self._bar_time(self.step_idx)
        step_trade_log_start = len(self.trade_log)
        entry_diag_before = dict(self._entry_diag)
        session_id = (
            int(self._session_ids_arr[self.step_idx]) if self._session_ids_arr is not None else 0
        )
        self._last_realized_close_pnl = None
        self._last_realized_close_r = None
        self._pending_realized_r = None
        self._opened_this_step = False
        self._trigger_requested_this_step = False

        next_market_idx = self.step_idx + 1
        market_fill_idx = min(next_market_idx, self._n_bars - 1)
        fill_idx = self.step_idx + 1
        if self.fill_latency_bars > 0:
            target_ny_pos = self._ny_pos + 1 + self.fill_latency_bars
            fill_idx = (
                int(self._ny_indices[target_ny_pos])
                if target_ny_pos < len(self._ny_indices)
                else self._n_bars
            )
        action_fill_time = self._bar_time(fill_idx) if fill_idx < self._n_bars else bar_time
        entry_fill_time = (
            self._bar_time(market_fill_idx)
            if market_fill_idx < self._n_bars
            else bar_time + pd.Timedelta(minutes=self._minutes_per_step)
        )
        fill_time = (
            self._bar_time(next_market_idx) + pd.Timedelta(minutes=self._minutes_per_step)
            if next_market_idx < self._n_bars
            else bar_time + pd.Timedelta(minutes=self._minutes_per_step)
        )
        self._active_step_execution = None
        if self.randomized_fill_latency_ms > 0:
            execution_order_time = pd.Timestamp(entry_fill_time)
            quote, execution_meta = self._resolve_randomized_tick_quote(execution_order_time)
            self._active_step_execution = {"quote": quote, "metadata": execution_meta}
            entry_fill_bid, entry_fill_ask = quote.bid, quote.ask
            entry_fill_time = quote.tick_time
            fill_time = quote.tick_time

        fill_bid, fill_ask, _ = self._progress_market(
            bar, feat_row, bar_time, session_id, market_fill_idx
        )
        if self._active_step_execution is not None:
            quote = self._active_step_execution["quote"]
            fill_bid, fill_ask = quote.bid, quote.ask
        elif market_fill_idx < self._n_bars:
            entry_fill_bid, entry_fill_ask = self._bar_open_quote(self._bar_at(market_fill_idx))
        else:
            entry_fill_bid, entry_fill_ask = fill_bid, fill_ask

        pending_order = self._pending_entry_order
        pending_due = bool(
            pending_order is not None and int(pending_order["fill_idx"]) == market_fill_idx
        )
        (
            entry_machine,
            entry_state_for_action,
            discrete_action,
            risk_frac,
            rr_ratio,
            _tp_mode,
            entry_candidate,
        ) = self._prepare_action_intent(
            action, feat_row, bar_time, session_id, fill_idx, pending_order
        )
        done, truncated, session_blocked, peak_dd_block = self._enforce_step_constraints(
            bar,
            feat_row,
            bar_time,
            session_id,
            fill_bid,
            fill_ask,
            fill_time,
            pending_due,
            entry_fill_bid,
            entry_fill_ask,
            entry_fill_time,
            market_fill_idx,
            entry_machine,
        )
        self._execute_step_action(
            action,
            discrete_action,
            risk_frac,
            rr_ratio,
            bar,
            feat_row,
            bar_time,
            session_id,
            fill_idx,
            action_fill_time,
            fill_bid,
            fill_ask,
            fill_time,
            entry_fill_bid,
            entry_fill_ask,
            entry_fill_time,
            market_fill_idx,
            entry_candidate,
            entry_machine,
            done,
            pending_due,
            session_blocked,
            peak_dd_block,
        )
        done, valuation_time, _pnl_step, reward, reward_parts = self._record_step_accounting(
            bar, feat_row, bar_time, step_trade_log_start, done, truncated
        )
        entry_funnel = self._update_entry_funnel(feat_row, entry_machine, entry_diag_before)
        obs, close_reason, info = self._build_step_info(
            bar,
            feat_row,
            bar_time,
            valuation_time,
            reward_parts,
            entry_funnel,
            entry_machine,
            entry_state_for_action,
            done,
        )
        observation, final_reward, terminated, truncated, final_info = self._resolve_step_end(
            action,
            discrete_action,
            risk_frac,
            entry_machine,
            valuation_time,
            close_reason,
            reward,
            reward_parts,
            obs,
            info,
            done,
            truncated,
        )
        return SimulationTransition(
            observation,
            float(final_reward),
            bool(terminated),
            bool(truncated),
            final_info,
        )

    def _progress_market(
        self,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
        session_id: int,
        fill_idx: int,
    ) -> tuple[float, float, None]:
        """Advance market state for the current bar.

        Tracks liquidity-level crossings, updates eval-mode session
        bookkeeping, marks open positions to market, computes the latency-shifted
        fill quote, and force-closes positions at session boundaries when
        ``block_overnight`` is enabled.

        Returns ``(fill_bid, fill_ask, None)`` — ``None`` is retained for
        API compatibility; the caller in :meth:`step` does not consume the
        latency-shifted bar.
        """
        price = float(bar.close)
        for level_name, level_value in (
            ("london_high", feat_row.get("london_high", float("nan"))),
            ("asian_high", feat_row.get("asian_high", float("nan"))),
            ("london_low", feat_row.get("london_low", float("nan"))),
            ("asian_low", feat_row.get("asian_low", float("nan"))),
        ):
            level_value = float(level_value)
            key = (session_id, level_name)
            if key in self._level_crosses or not np.isfinite(level_value):
                continue
            crossed = level_name.endswith("high") and price > level_value
            crossed = crossed or (level_name.endswith("low") and price < level_value)
            if crossed:
                self._level_crosses[key] = bar_time

        self.all_sessions.add(session_id)
        if self.prev_session is not None and session_id != self.prev_session:
            self.account.reset_daily()
        self.prev_session = session_id

        bid, ask = self._bar_quote(bar)
        if self.position is not None:
            self.broker.mark_to_market(self.account, self.position, (bid, ask))

        if fill_idx < self._n_bars:
            fill_bid, fill_ask = self._bar_quote(self._bar_at(fill_idx))
        else:
            fill_bid, fill_ask = bid, ask

        if self.block_overnight and self.position is not None and self._is_ny_session_boundary():
            eod_quote = (bid, ask)
            eod_fill_time = bar_time + pd.Timedelta(minutes=self._minutes_per_step)
            eod_deadline = eod_fill_time
            execution_meta: dict[str, Any] = {}
            if (
                getattr(self, "tick_execution_required", False)
                or getattr(self, "randomized_fill_latency_ms", 0.0) > 0
            ):
                if self._tickbook is None:
                    raise TickExecutionUnavailable("tick-only execution requires a TickBook")
                execution = resolve_tick_execution_by_deadline(
                    self._tickbook,
                    bar_time,
                    eod_deadline,
                    self.np_random,
                    max_latency_ns_exclusive=int(self.randomized_fill_latency_ms * 1_000_000),
                    max_quote_wait_ns=self.max_tick_quote_wait_ms * 1_000_000,
                )
                if execution is None:
                    raise TickExecutionUnavailable(
                        "no executable tick between order submission and scheduled session close "
                        f"{eod_deadline} within {self.max_tick_quote_wait_ms}ms"
                    )
                eod_quote = (execution.bid, execution.ask)
                eod_fill_time = execution.tick_time
                execution_meta = {
                    "order_submission_time": execution.order_time,
                    "execution_order_time": execution.order_time,
                    "execution_arrival_time": execution.arrival_time,
                    "execution_latency_ns": execution.latency_ns,
                    "actual_tick_time": execution.tick_time,
                    "fill_quote_source": "tick",
                }
            elif self._active_step_execution is not None:
                execution = self._active_step_execution["quote"]
                eod_quote = (execution.bid, execution.ask)
                eod_fill_time = execution.tick_time
            pnl, fill_price = self._close_at_quote(
                self.position,
                eod_quote,
                tick_fill=bool(execution_meta) or self._active_step_execution is not None,
            )
            record = {
                "type": "eod_close",
                "pnl": pnl,
                "price": fill_price,
                "reason": "session_end",
                "bar": self.step_idx,
                "time": bar_time,
                "order_submission_time": bar_time,
                "fill_time": eod_fill_time,
                "equity": self.account.equity,
                **execution_meta,
            }
            self._stamp_exit(record, fill_price)
            self.trade_log.append(record)
            self.position = None
            self.sessions_with_trades.add(session_id)
            self._note_close(pnl)

        return fill_bid, fill_ask, None

    def _handle_repeated_payout(self, event_time: Any) -> bool:
        """Record a flat realized payout success; reset capital only for continued eval."""
        if not self.repeated_payout_policy or self.position is not None:
            return False
        decision = evaluate_payout(
            self.account.balance,
            self.trading_policy.operating_balance,
            self.trading_policy.profit_target_balance,
        )
        if not decision.due:
            return False
        payout = decision.amount
        event = {
            "type": "payout_reset",
            "time": event_time,
            "cycle": int(self.cycle_count),
            "payout": payout,
            "balance_before_reset": float(self.account.balance),
            "operating_balance": float(self.trading_policy.operating_balance),
        }
        self.cumulative_payouts += payout
        self.cycle_successes += 1
        self.payout_events.append(dict(event))
        self.cycle_events.append({**event, "outcome": "success"})
        self.trade_log.append(dict(event))
        if self.episodic:
            return True
        self._payout_withdrawn_this_step = payout
        self.cycle_count += 1
        self.account.reset_operating_balance(self.trading_policy.operating_balance)
        return False

    def _calculate_reward(
        self,
        pnl_step: float,
        bar: BarView,
        feat_row: FeatureRow | pd.Series,
        done: bool,
        truncated: bool,
    ) -> float:
        """Compute the step reward from PnL and market state.

        Delegates to the active reward implementation (DSR or composite)
        with the appropriate kwargs.
        """
        daily_loss = float(self.account.canonical_daily_loss())
        # Elapsed session time in minutes. Derived from the bar spacing rather
        # than assumed, because the sweep time-decay term is a linear penalty
        # in this value and a wrong factor dominates the whole reward.
        minutes_since_open = max(0, self.step_idx - self.ny_session_start_idx) * (
            self._minutes_per_step
        )

        reward_kwargs: dict[str, Any] = {
            "daily_loss": daily_loss,
            "daily_loss_limit": self.guardrails.daily_loss_limit,
            "soft_daily_loss_limit": self.guardrails.soft_daily_loss_limit,
            "loss_from_initial": float(self.account.loss_from_initial()),
            "soft_max_loss_limit": self.guardrails.soft_max_loss_limit,
            "max_loss_limit": self.guardrails.max_loss_limit,
            "trailing_dd": float(self.account.trailing_drawdown_pct()),
            "soft_trailing_dd_limit": self.guardrails.soft_trailing_dd_limit,
            "trailing_dd_limit": self.guardrails.trailing_dd_limit,
            "initial_balance": self.initial_balance,
            # Year-fail is terminated (done) with truncated=False.
            "breach": bool(done and not truncated),
            "realized_close_pnl": self._last_realized_close_pnl,
            "realized_r": self._last_realized_close_r,
            "equity": float(self.account.equity),
        }

        if self.use_sweep_reward and isinstance(self.reward_fn, CompositeReward):
            position_changed = (
                self.position is not None
                and len(self.trade_log) > 0
                and self.trade_log[-1].get("type") in ("open", "close")
            )
            reward_kwargs.update(
                {
                    "cost": 0.0,
                    "price": float(bar.close),
                    "london_high": float(feat_row.get("london_high", float("nan"))),
                    "london_low": float(feat_row.get("london_low", float("nan"))),
                    "asian_high": float(feat_row.get("asian_high", float("nan"))),
                    "asian_low": float(feat_row.get("asian_low", float("nan"))),
                    "minutes_since_open": float(minutes_since_open),
                    "position_changed": bool(position_changed),
                }
            )

        if self.strategy_actions and isinstance(self.reward_fn, CompositeReward):
            if getattr(self.reward_fn, "strategy_reward", None) is not None:
                pos_dir = 0
                in_ifvg = False
                if self.position is not None:
                    pos_dir = int(self.position.direction)
                    in_ifvg = (
                        float(feat_row.get("price_in_ifvg_bull", 0.0)) > 0
                        if pos_dir == 1
                        else float(feat_row.get("price_in_ifvg_bear", 0.0)) > 0
                    )
                reward_kwargs["strategy_context"] = {
                    "position_changed": bool(self._opened_this_step),
                    "direction": pos_dir,
                    "in_ifvg": in_ifvg,
                    "manipulation_active": (
                        float(feat_row.get("po3_manipulation_active", 0.0)) > 0
                    ),
                    "manipulation_end": (float(feat_row.get("po3_manipulation_end", 0.0)) > 0),
                    "distribution_phase": (float(feat_row.get("po3_distribution", 0.0)) > 0),
                    "liquidity": (
                        self.strategy.entry_liquidity(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "entry_liquidity")
                        else True
                    ),
                    "in_gap": (
                        self.strategy.entry_gap(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "entry_gap")
                        else in_ifvg
                    ),
                    "entry_setup": (
                        self.strategy.entry_setup(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "entry_setup")
                        else False
                    ),
                    "distribution_after_ifvg": (
                        self.strategy.distribution_ready(feat_row, pos_dir)
                        if pos_dir != 0 and hasattr(self.strategy, "distribution_ready")
                        else False
                    ),
                    "sweep_high": float(feat_row.get("sweep_high", 0.0)) > 0,
                    "sweep_low": float(feat_row.get("sweep_low", 0.0)) > 0,
                    "sweep_high_reclaimed": float(feat_row.get("sweep_high_reclaimed", 0.0)) > 0,
                    "sweep_low_reclaimed": float(feat_row.get("sweep_low_reclaimed", 0.0)) > 0,
                    "bos_up": float(feat_row.get("bos_up", 0.0)) > 0,
                    "bos_down": float(feat_row.get("bos_down", 0.0)) > 0,
                }

        return float(self.reward_fn(pnl_step, **reward_kwargs))

    def _resolve_ny_indices(self, bars: pd.DataFrame) -> np.ndarray[Any, Any]:
        """Absolute indices of NY tradable bars, or all bars if none match."""
        if "session" in bars.columns:
            mask = bars["session"].eq("ny").to_numpy()
        else:
            mask = ny_session_mask(pd.DatetimeIndex(bars.index)).to_numpy()
        ny = np.flatnonzero(mask)
        if ny.size == 0:
            if self.strict_research_eligibility:
                raise ValueError("Strict research eligibility: no eligible NY session bars")
            ny = np.arange(len(bars), dtype=int)
        return ny

    def _is_ny_session_boundary(self) -> bool:
        """True on the last NY bar of a session (next NY index jumps)."""
        if self._ny_pos + 1 >= len(self._ny_indices):
            # The final eligible bar is also the end of the available session.
            return bool(len(self._ny_indices))
        return int(self._ny_indices[self._ny_pos + 1]) > int(self.step_idx) + 1

    def _advance_ny_step(self) -> None:
        """Move to the next NY bar, replaying skipped bars on an overnight hold."""
        cur = int(self.step_idx)
        if self._ny_pos + 1 >= len(self._ny_indices):
            self.step_idx = cur + 1
            self._ny_pos += 1
            return
        nxt = int(self._ny_indices[self._ny_pos + 1])
        if nxt > cur + 1 and self.position is not None and not self.block_overnight:
            bar = self._bar_at(cur)
            bid, ask = self._bar_quote(bar)
            self._apply_position_guards(
                bar,
                self._bar_time(cur),
                bid,
                ask,
                eod=True,
                feat_row=self._feature_row_at(cur),
                bar_idx=cur,
                fill_time=self._bar_time(cur) + pd.Timedelta(minutes=self._minutes_per_step),
            )
            self._replay_skipped_bars(cur + 1, nxt)
        self._ny_pos += 1
        self.step_idx = nxt

    def _replay_skipped_bars(self, start: int, end: int) -> None:
        """Mark-to-market and check SL/TP/age on bars the agent does not step."""
        for i in range(start, end):
            if self.position is None:
                break
            bar = self._bar_at(i)
            bid, ask = self._bar_quote(bar)
            self._check_sl_tp(bar, bid, ask, bar_idx=i)
            if self.position is None:
                break
            self.broker.mark_to_market(self.account, self.position, (bid, ask))
            self._apply_position_guards(
                bar,
                self._bar_time(i),
                bid,
                ask,
                eod=False,
                feat_row=self._feature_row_at(i),
                bar_idx=i,
                fill_time=self._bar_time(i) + pd.Timedelta(minutes=self._minutes_per_step),
            )
