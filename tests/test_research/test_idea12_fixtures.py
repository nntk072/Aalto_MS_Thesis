"""Numerical and causal checks for the synthetic Idea 1_2 illustrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "docs/decisions/fixtures/idea12_synthetic_fixtures.json"
)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, object]:
    return cast(dict[str, object], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def test_confirmed_long_example_has_causal_signal_submit_fill_order(
    fixtures: dict[str, object],
) -> None:
    example = fixtures["fixture_a_confirmed_long"]
    assert isinstance(example, dict)
    bars = example["bars"]
    assert isinstance(bars, list)
    assert [bar["bar"] for bar in bars] == [19, 20, 21, 22]
    assert bars[1]["low"] < example["reference"]["pivot_price"] < bars[1]["close"]
    assert bars[2]["close"] > example["fvg"]["high"]
    assert example["causal_order"][-3:] == [
        "IFVG confirmation close bar 21",
        "submission after bar 21 close",
        "fill at bar 22 open",
    ]
    assert (example["target"] - example["entry"]) / (
        example["entry"] - example["stop"]
    ) == pytest.approx(example["rr"])


def test_idea2_example_uses_completed_session_and_correct_rr(
    fixtures: dict[str, object],
) -> None:
    example = fixtures["fixture_b_idea2_gate"]
    assert isinstance(example, dict)
    session = example["previous_completed_session"]
    assert isinstance(session, dict)
    assert (session["high"] + session["low"]) / 2 == session["midpoint"]
    assert example["decision_close"] > session["midpoint"]
    assert example["new_long_eligible"] is False
    trade = example["illustrative_trade"]
    assert isinstance(trade, dict)
    assert (trade["target"] - trade["entry"]) / (trade["entry"] - trade["stop"]) == pytest.approx(
        trade["rr"]
    )
    alternate = example["alternate_stop_check"]
    assert isinstance(alternate, dict)
    assert (alternate["target"] - alternate["entry"]) / (
        alternate["entry"] - alternate["stop"]
    ) == pytest.approx(alternate["rr"])


def test_optional_retest_is_later_and_timing_remains_separate(
    fixtures: dict[str, object],
) -> None:
    example = fixtures["fixture_a_confirmed_long"]
    assert isinstance(example, dict)
    retest = example["optional_retest_example"]
    assert isinstance(retest, dict)
    assert retest["touch_bar"] > 21
    assert retest["zone"][0] <= retest["touch_price"] <= retest["zone"][1]
    assert retest["earliest_fill_bar"] > retest["touch_bar"]
