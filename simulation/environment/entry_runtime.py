# ruff: noqa: F403, F405
from __future__ import annotations

from .runtime_dependencies import *


class EntryRuntimeMixin(RuntimeMixinBase):
    def _prior_open_payload(self) -> dict[str, float | str] | None:
        """Detector fields for the open that happened on this step."""
        if not self._opened_this_step or not self.trade_log:
            return None
        row = self.trade_log[-1]
        if row.get("type") != "open":
            return None

        def _num(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        def _text(value: Any) -> str:
            if value is None:
                return ""
            return str(value)

        return {
            "stop_u": _num(row.get("stop_u")),
            "target_u": _num(row.get("target_u")),
            "sl_index": _num(row.get("sl_index")),
            "n_sl": _num(row.get("n_sl")),
            "tp_index": _num(row.get("tp_index")),
            "n_tp": _num(row.get("n_tp")),
            "planned_rr": _num(row.get("planned_rr")),
            "direction": _num(row.get("direction")),
            "exit_mode": _text(row.get("exit_mode")),
            "sl_ref": _text(row.get("sl_ref")),
            "tp_ref": _text(row.get("tp_ref")),
        }

    def _decision_fields(
        self,
        placed: Any,
        *,
        entry_price: float,
        risk_frac: float,
        entry_equity: float,
        feat_row: FeatureRow | pd.Series,
        bar_time: Any,
    ) -> dict[str, Any]:
        """Action coordinates and context for one strategy open."""
        from market_data.transforms.sessions import get_session
        from trading.orders.entries import reference_timeframe

        target = placed.tp_price
        stop_px = placed.sl_price
        target_distance = (
            abs(float(target) - float(entry_price)) if target is not None else float("nan")
        )
        stop_distance = (
            abs(float(entry_price) - float(stop_px)) if stop_px is not None else float("nan")
        )
        if self._bundle_arrays is not None and self._bundle_has_session:
            code = int(self._bundle_arrays["bar_session"][self.step_idx])
            label = self._session_labels[code] if code >= 0 else ""
            session = str(label) if label else ""
        elif self.bars is not None and "session" in self.bars.columns:
            label = self.bars["session"].iloc[self.step_idx]
            session = str(label) if pd.notna(label) and str(label) else ""
        else:
            try:
                session = get_session(pd.Timestamp(bar_time))
            except (TypeError, ValueError):
                session = ""
        tp_legs: list[dict[str, Any]] = []
        fraction_index = 0
        for leg_index, (price, reference) in enumerate(
            (
                (placed.tp1_price, placed.tp1_ref),
                (placed.tp2_price, placed.tp2_ref),
                (placed.tp3_price, placed.tp3_ref),
            ),
            start=1,
        ):
            if price is None:
                continue
            fractions = placed.tp_lot_fractions
            fraction = (
                float(fractions[fraction_index])
                if fraction_index < len(fractions)
                else float("nan")
            )
            fraction_index += 1
            tp_legs.append(
                {
                    "leg": leg_index,
                    "executed_price": float(price),
                    "reference": str(reference),
                    "lot_fraction": fraction,
                }
            )
        return {
            **self._decision_action,
            "sl_timeframe": reference_timeframe(str(placed.sl_ref)),
            "sl_index": int(placed.sl_index),
            "tp_timeframe": (
                "" if placed.exit_mode == "ema_21" else reference_timeframe(str(placed.tp_ref))
            ),
            "tp_index": float("nan") if int(placed.tp_index) < 0 else int(placed.tp_index),
            "stop_distance": stop_distance,
            "target_distance": target_distance,
            "tp_references": [asdict(reference) for reference in placed.tp_references],
            "tp_legs": tp_legs,
            # risk_cash is set unconditionally on the open row from the filled
            # lots, not here: reporting the requested budget let risk_cash read
            # 500.0 on trades that really risked $0.82, because min_lot and
            # max_loss_cap had clamped lots below the request.
            "session": session,
            "volatility_regime": self._volatility_regime(),
            "trend_regime": self._trend_regime(feat_row),
        }

    def _volatility_regime(self) -> str:
        """Causal tertile of ``atr_5`` over bars already seen in this episode."""
        if self._bundle_arrays is not None:
            values = self._bundle_arrays["atr_5"][: self.step_idx + 1]
            finite = values[np.isfinite(values)]
            if finite.size < 3:
                return ""
            current = float(finite[-1])
            low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
            if current <= float(low):
                return "low"
            return "high" if current >= float(high) else "mid"
        if "atr_5" not in self._features_col_to_idx:
            return ""
        end = self.step_idx + 1
        if end <= 0:
            return ""
        values = self._features_arr[:end, self._features_col_to_idx["atr_5"]]
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            return ""
        current = float(finite[-1])
        low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
        if current <= float(low):
            return "low"
        if current >= float(high):
            return "high"
        return "mid"

    def _trend_regime(self, feat_row: FeatureRow | pd.Series) -> str:
        if "htf_day_bias" not in feat_row.index:
            return ""
        try:
            bias = float(feat_row.get("htf_day_bias", np.nan))
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(bias):
            return ""
        if bias > 0.0:
            return "up"
        if bias < 0.0:
            return "down"
        return "flat"

    def _min_sl_distance(self, feat_row: FeatureRow | pd.Series) -> float:
        """Minimum allowed |entry - SL| in price points."""
        atr_val = feat_row.get("atr_5", feat_row.get("atr", np.nan))
        atr_f = float(atr_val) if atr_val is not None else float("nan")
        atr = atr_f if np.isfinite(atr_f) and atr_f > 0 else 0.0
        return float(max(float(self.min_sl_points), float(self.min_sl_atr_mult) * atr))

    def _filter_sl_candidates(
        self,
        *,
        direction: int,
        entry_price: float,
        candidates: list[tuple[str, float]],
        min_dist: float,
    ) -> list[tuple[str, float]]:
        from trading.orders.entries import filter_sl_candidates

        return filter_sl_candidates(
            direction=direction,
            entry_price=entry_price,
            candidates=candidates,
            min_dist=min_dist,
        )

    def _max_tp_here(self) -> float:
        """Reachable take-profit distance at the current bar, or NaN."""
        if self._max_tp_dist is None:
            return float("nan")
        i = self.step_idx
        if i < 0 or i >= len(self._max_tp_dist):
            return float("nan")
        return float(self._max_tp_dist[i])

    def _rr_fits(self, entry_price: float, sl_price: float, rr_ratio: float) -> bool:
        from trading.positions.tp_reach import rr_reaches

        return rr_reaches(entry_price, sl_price, rr_ratio, self._max_tp_here())

    def _resolve_trader_tp(
        self,
        *,
        direction: int,
        entry_price: float,
        sl_price: float,
        reward_fraction: float,
        feat_row: FeatureRow | pd.Series | None = None,
        max_tp_distance: float | None = None,
    ) -> float | None:
        from trading.orders.entries import dynamic_tp

        del feat_row
        tp, rejected = dynamic_tp(
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            reward_fraction=reward_fraction,
            max_tp_distance=max_tp_distance,
        )
        if rejected:
            return None
        return tp

    def _apply_direction_control(self, ctx: int, u: np.ndarray[Any, Any]) -> int:
        """Combine the agent's signed conviction with the heuristic side.

        ``u`` is the affine-mapped unit-interval action, where ``u[0] == 0.5``
        is neutral (PPO's deterministic mean) and the sign of
        ``u[0] - 0.5`` is the agent's side. The agent overrides the strategy
        only when its conviction is wider than
        ``direction_override_threshold``; otherwise ``ctx`` stands, so the
        heuristic side still governs at neutral. Returns ``-1``/``0``/``+1``.
        """
        conviction = float(u[0]) - 0.5
        if abs(conviction) < max(self.direction_override_threshold, 1e-6):
            return ctx if ctx in (-1, 1) else 0
        return 1 if conviction > 0.0 else -1

    def _finite(self, series: FeatureRow | pd.Series | BarView | None, name: str) -> float | None:
        if series is None or name not in series.index:
            return None
        value = float(series.get(name, np.nan))
        return value if np.isfinite(value) else None

    def _default_sl_mode(self, pos: Position | None = None) -> str:
        """State-machine default SL mode from market state.

        - ``fixed``: reference level not yet tested and TP is beyond the swing.
        - ``breakeven``: reference level has been tested.
        - ``trailing``: a new local swing formed during the move.
        """
        if pos is None:
            pos = self.position
        if pos is None:
            return "fixed"
        ref = pos.entry_sl_ref_price
        if ref is not None and pos.direction == 1:
            ref_taken = float(pos.mae) >= (pos.entry_price - float(ref))
        elif ref is not None and pos.direction == -1:
            ref_taken = float(pos.mae) >= (float(ref) - pos.entry_price)
        else:
            ref_taken = False
        tp_farther = False
        if pos.tp_price is not None and not np.isnan(pos.tp_price):
            if pos.direction == 1:
                swing = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_high",
                )
                tp_farther = swing is not None and float(pos.tp_price) > float(swing)
            elif pos.direction == -1:
                swing = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_low",
                )
                tp_farther = swing is not None and float(pos.tp_price) < float(swing)
        new_swing = False
        if pos.entry_last_swing_high is not None or pos.entry_last_swing_low is not None:
            cur_high = self._finite(
                self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                "last_swing_high",
            )
            cur_low = self._finite(
                self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                "last_swing_low",
            )
            if (
                pos.direction == 1
                and cur_high is not None
                and pos.entry_last_swing_high is not None
            ):
                new_swing = float(cur_high) > float(pos.entry_last_swing_high)
            elif (
                pos.direction == -1 and cur_low is not None and pos.entry_last_swing_low is not None
            ):
                new_swing = float(cur_low) < float(pos.entry_last_swing_low)
        if not ref_taken and tp_farther:
            return "fixed"
        if ref_taken:
            return "breakeven"
        if new_swing:
            return "trailing"
        return "fixed"

    def _default_tp_mode(self, pos: Position | None = None) -> str:
        """State-machine default TP mode. fixed -> breakeven -> trailing (monotonic)."""
        if pos is None:
            pos = self.position
        if pos is None:
            return "fixed"
        sl0 = pos.sl_initial_price if pos.sl_initial_price is not None else pos.sl_price
        if sl0 is None:
            return "fixed"
        risk_pts = abs(pos.entry_price - float(sl0))
        # breakeven: favorable excursion >= 0.5 * risk
        if risk_pts > 0 and float(pos.best_favorable) >= 0.5 * risk_pts:
            # Check for new swing -> trailing
            new_swing = False
            if pos.entry_last_swing_high is not None or pos.entry_last_swing_low is not None:
                cur_high = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_high",
                )
                cur_low = self._finite(
                    self._feature_row_at(self.step_idx) if self.step_idx >= 0 else None,
                    "last_swing_low",
                )
                if (
                    pos.direction == 1
                    and cur_high is not None
                    and pos.entry_last_swing_high is not None
                ):
                    new_swing = float(cur_high) > float(pos.entry_last_swing_high)
                elif (
                    pos.direction == -1
                    and cur_low is not None
                    and pos.entry_last_swing_low is not None
                ):
                    new_swing = float(cur_low) < float(pos.entry_last_swing_low)
            return "trailing" if new_swing else "breakeven"
        return "fixed"
