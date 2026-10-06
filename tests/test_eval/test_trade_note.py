"""Plain-language trade note shared by the PNG and HTML order charts."""

from __future__ import annotations

import math

import pytest

from quant_rl.eval.trade_note import (
    OPTION_CAP,
    build_trade_note,
    entry_reason,
    pretty_ref,
)

_OPEN_EMA = {
    "time": "2025-10-27 17:44:00+03:00",
    "sl_price": 25683.39,
    "tp_price": None,
    "sl_ref": "M5_last_swing_low",
    "tp_ref": "ema_21",
    "exit_mode": "ema_21",
    "level_type": "asian_low",
    "manipulation": "done",
    "strategy": "po3_ifvg",
    "sl_menu": [
        ["M5_last_swing_low", 25683.39],
        ["H1_last_swing_low", 25718.90],
        ["london_low", 25700.00],
    ],
    "tp_menu": [],
}
_CLOSE = {"time": "2025-10-27 17:48:00+03:00", "reason": "normal"}


def _note(open_row: dict[str, object], **over: object) -> list[str]:
    kwargs: dict[str, object] = {
        "entry_price": 25744.25,
        "exit_price": 25729.85,
        "direction": 1,
        "pnl": -21.53,
        "pnl_calc": -21.53,
        "duration_mins": 4,
        "duration_secs": 0,
        "lots": 1.49,
        "close_reason_detail": "normal",
    }
    kwargs.update(over)
    return build_trade_note(open_row, _CLOSE, **kwargs)  # type: ignore[arg-type]


def test_pretty_ref_maps_names_to_plain_language() -> None:
    assert pretty_ref("M5_last_swing_low") == "swing low (M5)"
    assert pretty_ref("H1_last_swing_high") == "swing high (H1)"
    assert pretty_ref("ema_21") == "EMA-21 exit"
    assert pretty_ref("asian_low") == "Asia session low"
    assert pretty_ref("H1_fvg_bull_high") == "fair-value gap (bullish)"
    assert pretty_ref("asian_low|H1_last_swing_low") == ("Asia session low + swing low (H1)")


def test_pretty_ref_is_safe_on_empty_and_nan() -> None:
    assert pretty_ref("") == ""
    assert pretty_ref(None) == ""
    assert pretty_ref(float("nan")) == ""
    assert pretty_ref(math.nan) == ""


def test_entry_reason_names_the_level_and_manipulation_state() -> None:
    text = entry_reason(_OPEN_EMA)
    assert text.startswith("Entry:")
    assert "swept Asia session low" in text
    assert 'manipulation "done"' in text


def test_entry_reason_mentions_distribution_phase() -> None:
    text = entry_reason({**_OPEN_EMA, "distribution_phase": 1.0})
    assert "distribution phase 1" in text


def test_entry_reason_degrades_when_columns_are_empty() -> None:
    text = entry_reason({})
    assert text == "Entry: strategy offered this side (no swept level)"
    assert entry_reason({"manipulation": ""}).strip() != ""


def test_note_has_direction_open_close_pnl_and_volume() -> None:
    joined = "\n".join(_note(_OPEN_EMA))
    assert "Long" in joined
    assert "17:44" in joined and "17:48" in joined
    assert "PnL -21.53" in joined
    assert "Volume 1.49" in joined
    assert "PnL logged -21.53 / calc -21.53" in joined


def test_ema_note_says_no_fixed_target() -> None:
    joined = "\n".join(_note(_OPEN_EMA))
    assert "no fixed target" in joined
    assert "TP options (0)  none - exit on EMA-21" in joined


def test_note_lists_options_and_marks_the_chosen_one() -> None:
    joined = "\n".join(_note(_OPEN_EMA))
    assert "SL options (3)" in joined
    assert "* swing low (M5) 25683.39" in joined


def test_option_line_is_capped() -> None:
    menu = [["london_low", 25700.0 - i] for i in range(OPTION_CAP + 4)]
    menu[0] = ["M5_last_swing_low", 25683.39]
    joined = "\n".join(_note({**_OPEN_EMA, "sl_menu": menu}))
    assert "more)" in joined or "(+" in joined
    assert joined.count(" 257") <= OPTION_CAP + 1


def test_structural_note_shows_tp_and_rr() -> None:
    row = {
        **_OPEN_EMA,
        "tp_price": 25876.83,
        "tp_ref": "H1_last_swing_low",
        "exit_mode": "structural",
        "tp_menu": [["H1_last_swing_low", 25876.83], ["asian_low", 25840.10]],
    }
    joined = "\n".join(_note(row))
    assert "TP 25876.83" in joined
    assert "TP options (2)" in joined


def test_short_direction_is_labelled() -> None:
    joined = "\n".join(_note(_OPEN_EMA, direction=-1))
    assert "Short" in joined


@pytest.mark.parametrize("bad", [None, "", "not-a-menu", 3, float("nan")])
def test_menu_parsing_tolerates_junk(bad: object) -> None:
    joined = "\n".join(_note({**_OPEN_EMA, "sl_menu": bad}))
    assert joined  # never raises, still produces the note
    assert "SL options" not in joined
