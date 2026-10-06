"""Detector for the selective mode-mean prior. It does not touch the reward."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.eval.decision_diversity import (
    active_prior_names,
    candidate_rank,
    risk_mean_collapsed,
)

_N = 40


def _book(n: int, **cols: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "sl_ref": "swing_low",
        "tp_ref": "swing_high",
        "exit_mode": "structural",
        "direction": 1,
        "stop_u": 0.25,
        "target_u": 0.35,
        "n_sl": 3,
        "n_tp": 3,
        "sl_index": 1,
        "tp_index": 1,
    }
    row.update(cols)
    data: dict[str, list[object]] = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            data[key] = list(value)
        else:
            data[key] = [value] * n
    return pd.DataFrame(data)


def _spread(n: int) -> list[float]:
    return [float(x) for x in np.linspace(0.0, 1.0, n)]


def test_risk_floor_turns_on_without_opens() -> None:
    assert risk_mean_collapsed(0.0, 1000)
    assert risk_mean_collapsed(0.05, 1000)
    assert not risk_mean_collapsed(0.5, 5000)
    assert not risk_mean_collapsed(0.0, 10)


def test_candidate_rank_ends_are_zero_and_one() -> None:
    assert candidate_rank(0, 5) == 0.0
    assert candidate_rank(4, 5) == 1.0
    assert candidate_rank(0, 1) == 0.0


def test_single_candidate_does_not_turn_stop_or_target_on() -> None:
    stop = _book(_N, n_sl=1, sl_index=0, stop_u=1.0)
    assert "stop" not in active_prior_names(stop)
    target = _book(_N, n_tp=1, tp_index=0, target_u=1.0)
    assert "target" not in active_prior_names(target)
    ema = _book(_N, exit_mode="ema_21", n_tp=8, tp_index=7, target_u=1.0)
    assert "target" not in active_prior_names(ema)


def test_short_window_returns_an_empty_set() -> None:
    parked = _book(20, n_sl=5, sl_index=4, stop_u=1.0, exit_mode="ema_21")
    assert active_prior_names(parked) == set()


def test_far_rank_turns_stop_on_and_family_share_does_not() -> None:
    far = _book(_N, n_sl=5, sl_index=4, stop_u=_spread(_N), target_u=_spread(_N))
    assert active_prior_names(far) == {"stop"}
    crowded = _book(
        _N,
        sl_ref="H1_swing_high",
        n_sl=3,
        sl_index=[0, 1] * (_N // 2),
        stop_u=_spread(_N),
        target_u=_spread(_N),
    )
    assert "stop" not in active_prior_names(crowded)


def test_target_matches_stop() -> None:
    parked = _book(_N, n_tp=4, tp_index=1, target_u=0.9, stop_u=_spread(_N))
    assert active_prior_names(parked) == {"target"}
    far = _book(_N, n_tp=5, tp_index=4, target_u=_spread(_N), stop_u=_spread(_N))
    assert active_prior_names(far) == {"target"}


def test_exit_and_direction_floors() -> None:
    mixed_exit = ["ema_21"] * 75 + ["structural"] * 25
    ema = _book(100, exit_mode=mixed_exit, direction=[1, -1] * 50, stop_u=_spread(100))
    assert "exit" not in active_prior_names(ema)
    collapsed_exit = _book(
        60,
        exit_mode="ema_21",
        direction=[1, -1] * 30,
        stop_u=_spread(60),
        target_u=_spread(60),
    )
    assert active_prior_names(collapsed_exit) == {"exit"}

    book = [1] * 20 + [-1] * 53 + [0] * 27
    mixed = _book(100, direction=book, stop_u=_spread(100), target_u=_spread(100))
    assert "direction" not in active_prior_names(mixed)

    tiny = _book(50, direction=[-1, -1] + [0] * 48, stop_u=_spread(50), target_u=_spread(50))
    assert "direction" not in active_prior_names(tiny)

    one_side = _book(
        100,
        direction=[-1] * 95 + [1] * 5,
        exit_mode=["ema_21", "structural"] * 50,
        stop_u=_spread(100),
        target_u=_spread(100),
    )
    assert active_prior_names(one_side) == {"direction"}
