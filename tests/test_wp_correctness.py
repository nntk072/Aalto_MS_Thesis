"""Focused regressions for the October 12 implementation audit."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from quant_rl.envs.trading_env import TradingEnv
from quant_rl.evaluation.bootstrap_ci import metrics_with_ci
from quant_rl.models.ppo_policy import mode_index


@pytest.mark.parametrize(
    ("width", "flags", "expected"),
    [
        (8, {"has_sl_mode": True, "has_multi_tp": True}, 1),
        (9, {"has_sl_mode": True, "has_tp_mode": True, "has_multi_tp": True}, 1),
        (
            11,
            {"has_sl_mode": True, "has_tp_mode": True, "has_multi_tp": True, "has_simplex": True},
            1,
        ),
        (
            12,
            {
                "has_direction": True,
                "has_sl_mode": True,
                "has_tp_mode": True,
                "has_multi_tp": True,
                "has_simplex": True,
            },
            2,
        ),
    ],
)
def test_stop_prior_uses_flags_not_action_width(
    width: int, flags: dict[str, bool], expected: int
) -> None:
    assert mode_index("stop", width, **flags) == expected


@pytest.mark.parametrize(
    ("side", "swing_low", "swing_high", "expected"),
    [(1, 98.0, 110.0, 97.5), (-1, 90.0, 102.0, 102.5)],
)
def test_trailing_stop_uses_protective_confirmed_swing(
    side: int, swing_low: float, swing_high: float, expected: float
) -> None:
    stub_env: Any = SimpleNamespace(sl_buffer_pts=0.5, _finite=lambda row, key: row.get(key))
    pos: Any = SimpleNamespace(
        sl_mode="trailing", direction=side, sl_price=95.0 if side == 1 else 105.0
    )
    TradingEnv._apply_trailing_stop(
        stub_env,
        pos,
        cast(Any, {"last_swing_low": swing_low, "last_swing_high": swing_high}),
    )
    assert pos.sl_price == pytest.approx(expected)


@pytest.mark.parametrize("side,prices", [(1, (110.0, 110.0, 120.0)), (-1, (90.0, 90.0, 80.0))])
def test_tp_reorder_enforces_strict_order_without_duplicates(
    side: int, prices: tuple[float, ...]
) -> None:
    pos: Any = SimpleNamespace(
        direction=side,
        tp1_price=prices[0],
        tp2_price=prices[1],
        tp3_price=prices[2],
        entry_price=100.0,
        tp_hit_mask=0,
    )
    TradingEnv._reorder_tp_levels_safe(cast(Any, None), pos)
    remaining = [pos.tp1_price, pos.tp2_price, pos.tp3_price]
    active = [price for price in remaining if price is not None]
    assert len(active) >= 1
    assert all(
        active[i] < active[i + 1] if side == 1 else active[i] > active[i + 1]
        for i in range(len(active) - 1)
    )


def test_bootstrap_mdd_does_not_invent_drawdown_on_rising_equity() -> None:
    equity = np.linspace(100, 130, 30)
    ci = metrics_with_ci(np.asarray(equity, dtype=np.float64), np.array([1.0]), n_boot=80)[
        "max_drawdown"
    ]
    assert ci.estimate == pytest.approx(0.0)
    assert ci.upper == pytest.approx(0.0)
