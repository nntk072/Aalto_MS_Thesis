"""Human-readable trade note for the per-trade order charts.

Both renderers (matplotlib PNG and plotly HTML) build their caption here so the
two always agree. Nothing here changes a price, a level, or the environment; it
only turns an open/close row into plain-language lines.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

# Cap the printed ladder so the note stays a few lines.
OPTION_CAP: int = 4
STAR: str = "*"

_SIDE_TF: tuple[str, ...] = ("M5", "M15", "H1")

_SIMPLE: dict[str, str] = {
    "last_swing_low": "swing low (M1)",
    "last_swing_high": "swing high (M1)",
    "asian_low": "Asia session low",
    "asian_high": "Asia session high",
    "london_low": "London session low",
    "london_high": "London session high",
    "prev_day_high": "previous day high",
    "prev_day_low": "previous day low",
    "prev_day_high_low": "previous day high/low",
    "ctx_prev_week_high": "previous week high",
    "ctx_prev_week_low": "previous week low",
    "sweep_high_level": "swept liquidity high",
    "sweep_low_level": "swept liquidity low",
    "smt_swing": "SMT swing",
    "sl_reference": "strategy stop reference",
    "ema_21": "EMA-21 exit",
}

_LEVEL_ENTRY: dict[str, str] = {
    "asian_low": "swept Asia session low",
    "asian_high": "swept Asia session high",
    "london_low": "swept London session low",
    "london_high": "swept London session high",
    "prev_day_high": "previous day high",
    "prev_day_low": "previous day low",
    "ctx_prev_week_high": "previous week high",
    "ctx_prev_week_low": "previous week low",
}

_MANIPULATION: dict[str, str] = {
    "done": 'manipulation "done" (phase completed)',
    "none": "no manipulation yet",
    "against": 'manipulation "against" (side not confirmed)',
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def pretty_ref(name: Any) -> str:
    """Raw level reference in plain language.

    ``asian_low|H1_last_swing_low`` becomes ``Asia session low + swing low (H1)``.
    An empty or ``nan`` input returns ``""``.
    """
    text = _text(name)
    if not text or text.lower() == "nan":
        return ""
    out: list[str] = []
    for part in (p.strip() for p in text.split("|")):
        if not part:
            continue
        if part in _SIMPLE:
            out.append(_SIMPLE[part])
            continue
        hit = next(
            (
                (tf, swing)
                for tf in _SIDE_TF
                for swing in ("last_swing_low", "last_swing_high")
                if part == f"{tf}_{swing}"
            ),
            None,
        )
        if hit is not None:
            out.append(f"swing {'low' if hit[1].endswith('low') else 'high'} ({hit[0]})")
            continue
        if "ifvg" in part:
            out.append(f"inverse gap ({'bullish' if 'bull' in part else 'bearish'})")
            continue
        if "fvg" in part:
            out.append(f"fair-value gap ({'bullish' if 'bull' in part else 'bearish'})")
            continue
        if part.startswith("dev"):
            out.append(f"deviation {part.split('_', 1)[-1]}")
            continue
        out.append(part.replace("_", " "))
    return " + ".join(out)


def entry_reason(open_row: Any) -> str:
    """Why the trade was taken, from columns already on the open row.

    ``enforce_gate`` is false in both idea configs, so this describes the setup
    the strategy offered, not a filter that blocked other trades.
    """
    parts: list[str] = []
    level = _LEVEL_ENTRY.get(_text(open_row.get("level_type")))
    if level is None:
        level = pretty_ref(open_row.get("level_type"))
    parts.append(level or "strategy offered this side (no swept level)")
    state = _text(open_row.get("manipulation")).lower()
    if state:
        parts.append(_MANIPULATION.get(state, f"manipulation {state}"))
    phase = _number(open_row.get("distribution_phase"))
    if phase is not None and phase > 0.0:
        parts.append(f"distribution phase {int(phase)}")
    return "Entry: " + ", ".join(parts)


def _menu_pairs(raw: Any) -> list[tuple[str, float]]:
    """Normalize a logged ``sl_menu`` / ``tp_menu`` cell into pairs."""
    if raw is None or isinstance(raw, (str, bytes, float, int, bool)):
        return []
    try:
        items: Iterable[Any] = list(raw)
    except TypeError:
        return []
    out: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            name = _text(item[0])
            price = _number(item[1])
            if name and price is not None:
                out.append((name, price))
    return out


def _format_options(
    menu: Sequence[tuple[str, float]],
    chosen_name: str,
    entry_price: float,
    label: str,
) -> str:
    """One line listing the ladder, ``*`` marking the agent's choice."""
    if not menu:
        return f"{label} options (0)  none"
    pick = next((i for i, (name, _) in enumerate(menu) if name == chosen_name), None)
    near = sorted(range(len(menu)), key=lambda i: abs(menu[i][1] - entry_price))
    if pick is None:
        shown = near[: OPTION_CAP + 1]
    else:
        rest = [i for i in near if i != pick][:OPTION_CAP]
        shown = [pick, *rest]
    body = " . ".join(
        (STAR + " " if i == pick else "")
        + (pretty_ref(menu[i][0]) or menu[i][0])
        + f" {menu[i][1]:.2f}"
        for i in shown
    )
    more = "" if len(shown) >= len(menu) else f"  (+{len(menu) - len(shown)} more)"
    return f"{label} options ({len(menu)})  {body}{more}"


def _clock(value: Any) -> str:
    text = _text(value)
    return text[11:16] if len(text) >= 16 else text


def build_trade_note(
    open_row: Any,
    close_row: Any,
    *,
    entry_price: float,
    exit_price: float,
    direction: int,
    pnl: float,
    pnl_calc: float,
    duration_mins: int,
    duration_secs: int,
    lots: float,
    close_reason_detail: str,
    extra_notes: Sequence[str] | None = None,
) -> list[str]:
    """Caption lines shown under both subplots of one trade chart."""
    dir_label = "Long" if direction == 1 else "Short" if direction == -1 else "Flat"
    lines = [
        f"{dir_label} - Open {_clock(open_row.get('time'))}"
        f" -> Close {_clock(close_row.get('time'))}"
        f" ({duration_mins}m{duration_secs:02d}s)"
        f" - PnL {pnl:+.2f} - Volume {lots:.2f} lots",
        entry_reason(open_row),
        f"PnL logged {pnl:+.2f} / calc {pnl_calc:+.2f} / closed: {close_reason_detail}",
    ]

    exit_mode = _text(open_row.get("exit_mode"))
    sl_ref = _text(open_row.get("sl_ref"))
    tp_ref = _text(open_row.get("tp_ref"))
    sl_price = _number(open_row.get("sl_price"))
    tp_price = _number(open_row.get("tp_price"))
    ema_exit = exit_mode == "ema_21" or tp_ref.lower() == "ema_21"

    stop = (
        f"SL {sl_price:.2f} {pretty_ref(sl_ref) or 'stop'}"
        if sl_price is not None
        else (f"SL {pretty_ref(sl_ref) or 'stop'}")
    )
    if ema_exit and tp_price is None:
        target = f"Exit {exit_price:.2f} EMA-21 exit (no fixed target)"
    elif tp_price is not None:
        target = f"TP {tp_price:.2f} {pretty_ref(tp_ref) or 'target'}"
    else:
        target = f"Exit {exit_price:.2f}"
    lines.append(f"{stop}  |  {target}")

    sl_menu = _menu_pairs(open_row.get("sl_menu"))
    tp_menu = _menu_pairs(open_row.get("tp_menu"))
    if sl_menu:
        lines.append(_format_options(sl_menu, sl_ref, entry_price, "SL"))
    if ema_exit:
        lines.append(
            f"TP options ({len(tp_menu)})  listed but not used (EMA-21 exit)"
            if tp_menu
            else "TP options (0)  none - exit on EMA-21"
        )
    elif tp_menu:
        lines.append(_format_options(tp_menu, tp_ref, entry_price, "TP"))

    for note in extra_notes or ():
        lines.append(f"note: {note}")
    return lines


__all__ = ["OPTION_CAP", "STAR", "build_trade_note", "entry_reason", "pretty_ref"]
