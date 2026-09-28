"""Decoded decision diversity, separate from raw action spread."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl.eval.decision_diversity import (
    candidate_rank,
    diversity_report,
    shannon_entropy,
    write_decision_artifacts,
)


def _open(**values: object) -> dict[str, object]:
    row: dict[str, object] = {
        "type": "open",
        "exit_mode": "structural",
        "planned_rr": 2.0,
        "sl_index": 0,
        "n_sl": 2,
        "tp_index": 0,
        "n_tp": 2,
        "stop_u": 0.5,
        "target_u": 0.5,
        "manipulation": "done",
        "sl_ref": "last_swing_low",
        "tp_ref": "prev_day_high",
    }
    row.update(values)
    return row


def test_same_reference_is_one_choice_despite_raw_spread() -> None:
    trades = pd.DataFrame(
        [
            _open(stop_u=0.51, sl_ref="last_swing_low"),
            _open(stop_u=0.49, sl_ref="last_swing_low"),
        ]
    )
    report = diversity_report(trades)
    assert report.sl_shares.index.tolist() == ["swing"]
    assert float(report.sl_shares.iloc[0]) == 1.0
    assert "unique sl families 1" in report.text


def test_entropy_of_one_label_and_a_balanced_split() -> None:
    assert shannon_entropy(pd.Series(["swing", "swing"])) == 0.0
    split = shannon_entropy(pd.Series(["swing", "sweep"]))
    assert np.exp(split) == 2.0


def test_manipulation_changes_the_stop_family() -> None:
    trades = pd.DataFrame(
        [
            _open(manipulation="against", sl_ref="sweep_low_level"),
            _open(manipulation="against", sl_ref="sweep_low_level"),
            _open(manipulation="done", sl_ref="last_swing_low"),
            _open(manipulation="done", sl_ref="ifvg_bull_low"),
        ]
    )
    report = diversity_report(trades)
    table = report.by_conditioner["manipulation"]
    against = table.loc["against"]
    done = table.loc["done"]

    def _share(row: pd.Series | pd.DataFrame, name: str) -> float:
        assert isinstance(row, pd.Series)
        if name not in row.index:
            return 0.0
        value = row[name]
        assert isinstance(value, (int, float))
        return float(value)

    assert _share(against, "sweep") > _share(against, "swing")
    assert _share(done, "sweep") == 0.0
    assert _share(done, "swing") > 0.0
    assert _share(done, "fvg") > 0.0


def test_nearest_targets_warn_and_a_spread_does_not() -> None:
    flat = diversity_report(pd.DataFrame([_open(planned_rr=1.0) for _ in range(8)]))
    spread = diversity_report(
        pd.DataFrame([_open(planned_rr=value) for value in (1.2, 2.0, 3.0, 5.0, 8.0)])
    )
    assert any(item.startswith("RR COLLAPSE") for item in flat.warnings)
    assert all(not item.startswith("RR COLLAPSE") for item in spread.warnings)


def test_rank_endpoints() -> None:
    assert candidate_rank(0, 1) == 0.0
    assert candidate_rank(3, 4) == 1.0
    assert np.isnan(candidate_rank(-1, 4))


def test_ema_exits_skip_planned_rr_and_stay_in_the_exit_share() -> None:
    only = diversity_report(
        pd.DataFrame([_open(exit_mode="ema_21", tp_ref="ema_21", tp_index=-1, planned_rr=np.nan)])
    )
    assert only.rr.empty
    assert float(only.exit_shares["ema_21"]) == 1.0
    mixed = diversity_report(
        pd.DataFrame(
            [
                _open(planned_rr=4.0, exit_mode="structural"),
                _open(exit_mode="ema_21", tp_ref="ema_21", tp_index=-1, planned_rr=np.nan),
            ]
        )
    )
    assert float(mixed.rr.median()) == 4.0
    assert set(mixed.exit_shares.index) == {"structural", "ema_21"}


def test_artifacts_write_the_core_figures(tmp_path) -> None:
    trades = pd.DataFrame(
        [
            _open(sl_ref="sweep_low_level", tp_ref="prev_day_high", stop_u=0.1, sl_index=0),
            _open(sl_ref="last_swing_low", tp_ref="london_high", stop_u=0.8, sl_index=1),
        ]
    )
    write_decision_artifacts(trades, tmp_path, dpi=72)
    for name in (
        "decision_diversity.txt",
        "sl_family.png",
        "tp_family.png",
        "planned_rr_hist.png",
        "sl_tp_rank_heatmap.png",
        "sl_by_manipulation.png",
        "stop_u_vs_index.png",
        "target_u_vs_index.png",
    ):
        assert (tmp_path / name).is_file()
