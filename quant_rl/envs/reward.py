"""Differential Sharpe Ratio (DSR) and PnL rewards.

Reference: Moody & Saffell (2001) "Learning to trade via direct reinforcement".
Reward = dS_t / dF_t · ΔF_t  (first-order approximation)

We also add:
- Soft penalty for FTMO daily-loss proximity
- Soft penalty for year max-loss proximity (loss_from_initial)
- Cost term (already captured in PnL but can be explicit)
- Hard breach → terminal negative reward
"""

from __future__ import annotations

import numpy as np

REWARD_PART_KEYS: tuple[str, ...] = (
    "pnl",
    "dsr",
    "soft_daily",
    "soft_year",
    "soft_trailing",
    "strategy",
    "breach",
    "sweep",
    "peak_dd",
)


def empty_reward_parts() -> dict[str, float]:
    """Zero split of one step reward. The keys sum to the returned reward."""
    return {key: 0.0 for key in REWARD_PART_KEYS}


def fit_reward_parts(parts: dict[str, float], raw: float, clipped: float) -> dict[str, float]:
    """Scale a split so it sums to the clipped reward the agent receives."""
    out = empty_reward_parts()
    out.update({key: float(value) for key, value in parts.items()})
    if raw != 0.0 and clipped != raw:
        scale = clipped / raw
        out = {key: float(value) * scale for key, value in out.items()}
    return out


def _soft_band_penalty(
    value: float,
    soft: float | None,
    hard: float,
    *,
    weight: float = 0.5,
) -> float:
    """Linear penalty in ``(soft, hard]``; 0 below soft."""
    soft_v = float(soft) if soft is not None and soft > 0.0 else 0.0
    if soft_v <= 0.0 or hard <= soft_v or value <= soft_v:
        return 0.0
    excess = (value - soft_v) / (hard - soft_v + 1e-9)
    return weight * float(np.clip(excess, 0.0, 1.0))


def _fresh_band_state() -> dict[str, bool]:
    return {"soft_daily": False, "soft_year": False, "soft_trailing": False}


def _charge_band_once(
    state: dict[str, bool],
    key: str,
    value: float,
    soft: float | None,
    hard: float,
    *,
    weight: float = 0.5,
) -> float:
    """Penalise the bar that enters the band. Later bars inside it pay nothing."""
    penalty = _soft_band_penalty(value, soft, hard, weight=weight)
    inside = penalty > 0.0
    entered = inside and not state[key]
    state[key] = inside
    return -penalty if entered else 0.0


class DSRReward:
    """Online differential Sharpe reward with FTMO soft penalties."""

    def __init__(self, eta: float = 0.01, *, breach_penalty: float = -1.0) -> None:
        self.eta = eta  # EMA damping for A and B estimates
        self.breach_penalty = float(breach_penalty)
        self._A: float = 0.0  # EMA of returns
        self._B: float = 0.0  # EMA of squared returns
        self._band_inside = _fresh_band_state()
        self.last_parts: dict[str, float] = empty_reward_parts()

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0
        self._band_inside = _fresh_band_state()

    def __call__(
        self,
        step_pnl: float,
        *,
        daily_loss: float = 0.0,
        daily_loss_limit: float = 5_000.0,
        soft_daily_loss_limit: float | None = 2_000.0,
        loss_from_initial: float = 0.0,
        soft_max_loss_limit: float | None = 5_000.0,
        max_loss_limit: float = 10_000.0,
        trailing_dd: float = 0.0,
        soft_trailing_dd_limit: float | None = 0.04,
        trailing_dd_limit: float = 0.07,
        initial_balance: float = 100_000.0,
        breach: bool = False,
        realized_close_pnl: float | None = None,  # noqa: ARG002
        equity: float | None = None,  # noqa: ARG002
    ) -> float:
        """Compute the DSR reward for one step.

        Parameters
        ----------
        step_pnl:
            Realised + unrealised P&L change for this bar.
        daily_loss, daily_loss_limit, soft_daily_loss_limit:
            Soft FTMO penalty ramps from the soft brick toward the hard limit.
        loss_from_initial, soft_max_loss_limit, max_loss_limit:
            Soft year-loss shaping from soft floor toward absolute max loss.
        trailing_dd, soft_trailing_dd_limit, trailing_dd_limit:
            Soft trailing (peak-to-trough) drawdown band toward the 7% hard cap.
        initial_balance:
            For normalisation.
        breach:
            Hard FTMO guardrail breached → large negative terminal reward.
        """
        if breach:
            penalty = self.breach_penalty
            self.last_parts = fit_reward_parts({"breach": penalty}, penalty, penalty)
            return penalty

        # Normalise P&L to relative return
        r = step_pnl / initial_balance

        # Update EMA estimates
        A_prev = self._A
        B_prev = self._B
        self._A = A_prev + self.eta * (r - A_prev)
        self._B = B_prev + self.eta * (r**2 - B_prev)

        denom = self._B - self._A**2
        if denom <= 1e-10:
            dsr = 0.0
        else:
            dsr = (self._B * (r - A_prev) - 0.5 * self._A * (r**2 - B_prev)) / (denom**1.5)

        # Soft brick → hard limit: linear penalty between soft and hard daily loss.
        soft = (
            float(soft_daily_loss_limit)
            if soft_daily_loss_limit is not None and soft_daily_loss_limit > 0.0
            else 0.8 * daily_loss_limit
        )
        soft_daily = _charge_band_once(
            self._band_inside, "soft_daily", daily_loss, soft, daily_loss_limit
        )
        soft_year = _charge_band_once(
            self._band_inside, "soft_year", loss_from_initial, soft_max_loss_limit, max_loss_limit
        )
        soft_trailing = _charge_band_once(
            self._band_inside,
            "soft_trailing",
            trailing_dd,
            soft_trailing_dd_limit,
            trailing_dd_limit,
        )
        raw = dsr + soft_daily + soft_year + soft_trailing
        clipped = float(np.clip(raw, -10.0, 10.0))
        self.last_parts = fit_reward_parts(
            {
                "dsr": dsr,
                "soft_daily": soft_daily,
                "soft_year": soft_year,
                "soft_trailing": soft_trailing,
            },
            raw,
            clipped,
        )
        return clipped


class PnLReward:
    """Realized-PnL reward with soft daily-loss penalty from 1% of equity.

    Used for Idea 1/2 trader-like training (``env.reward_mode: pnl``).
    Baseline Idea 3 keeps :class:`DSRReward`.
    """

    def __init__(
        self,
        *,
        soft_loss_frac: float = 0.01,
        dsr_weight: float = 0.0,
        breach_penalty: float = -1.0,
        soft_band_weight: float = 0.05,
    ) -> None:
        self.soft_loss_frac = soft_loss_frac
        self.dsr_weight = float(dsr_weight)
        self.breach_penalty = float(breach_penalty)
        self.soft_band_weight = float(soft_band_weight)
        self._dsr = DSRReward(breach_penalty=self.breach_penalty) if self.dsr_weight > 0 else None
        self._band_inside = _fresh_band_state()
        self.last_parts: dict[str, float] = empty_reward_parts()

    def reset(self) -> None:
        self._band_inside = _fresh_band_state()
        if self._dsr is not None:
            self._dsr.reset()

    def __call__(
        self,
        step_pnl: float,
        *,
        daily_loss: float = 0.0,
        daily_loss_limit: float = 5_000.0,
        soft_daily_loss_limit: float | None = 2_000.0,
        loss_from_initial: float = 0.0,
        soft_max_loss_limit: float | None = 5_000.0,
        max_loss_limit: float = 10_000.0,
        trailing_dd: float = 0.0,
        soft_trailing_dd_limit: float | None = 0.04,
        trailing_dd_limit: float = 0.07,
        initial_balance: float = 100_000.0,
        breach: bool = False,
        realized_close_pnl: float | None = None,
        equity: float | None = None,
    ) -> float:
        if breach:
            penalty = self.breach_penalty
            self.last_parts = fit_reward_parts({"breach": penalty}, penalty, penalty)
            return penalty

        parts = empty_reward_parts()
        if realized_close_pnl is not None:
            parts["pnl"] = float(realized_close_pnl) / initial_balance

        eq = float(equity) if equity is not None else initial_balance
        soft_start = (
            float(soft_daily_loss_limit)
            if soft_daily_loss_limit is not None and soft_daily_loss_limit > 0.0
            else self.soft_loss_frac * max(eq, 1.0)
        )
        parts["soft_daily"] = _charge_band_once(
            self._band_inside,
            "soft_daily",
            daily_loss,
            soft_start,
            daily_loss_limit,
            weight=self.soft_band_weight,
        )
        parts["soft_year"] = _charge_band_once(
            self._band_inside,
            "soft_year",
            loss_from_initial,
            soft_max_loss_limit,
            max_loss_limit,
            weight=self.soft_band_weight,
        )
        parts["soft_trailing"] = _charge_band_once(
            self._band_inside,
            "soft_trailing",
            trailing_dd,
            soft_trailing_dd_limit,
            trailing_dd_limit,
            weight=self.soft_band_weight,
        )

        if self._dsr is not None:
            self._dsr(
                step_pnl,
                daily_loss=daily_loss,
                daily_loss_limit=daily_loss_limit,
                soft_daily_loss_limit=soft_daily_loss_limit,
                loss_from_initial=loss_from_initial,
                soft_max_loss_limit=soft_max_loss_limit,
                max_loss_limit=max_loss_limit,
                trailing_dd=trailing_dd,
                soft_trailing_dd_limit=soft_trailing_dd_limit,
                trailing_dd_limit=trailing_dd_limit,
                initial_balance=initial_balance,
                breach=False,
            )
            for key, value in self._dsr.last_parts.items():
                parts[key] = parts.get(key, 0.0) + self.dsr_weight * float(value)

        raw = float(sum(parts.values()))
        clipped = float(np.clip(raw, -10.0, 10.0))
        self.last_parts = fit_reward_parts(parts, raw, clipped)
        return clipped
