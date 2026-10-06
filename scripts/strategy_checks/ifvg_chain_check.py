"""Liquidity sweep -> IFVG -> prior swing high: the Idea 2.1 chain, isolated.

This is the first test of the full chain the user described rather than any
single leg: take liquidity, let an inverse fair value gap form, and target the
swing high the move came from.

Two entry variants, since "respected" is ambiguous:
  * ``ifvg_noretest``  enter on the displacement bar itself, never waiting
  * ``ifvg_retest``    enter when price returns into the inverted gap and holds

Every trade is resolved bar-by-bar against a real stop and target, so the
result is R-multiples and win rate rather than a forward-return average.

Diagnostic only. Writes no training config and launches no run.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from edge_harness import (  # type: ignore[import-not-found]
    OUT,
    atr_series,
    build_ny_frame,
    displacement_flags,
    fvg_zones,
    load_bars,
    sweep_flags,
    swing_levels,
)

STOP_ATR = 0.5
MAX_HOLD = 180


def resolve(
    ny: pd.DataFrame,
    entry: np.ndarray[Any, Any],
    direction: np.ndarray[Any, Any],
    stop: np.ndarray[Any, Any],
    target: np.ndarray[Any, Any],
) -> pd.DataFrame:
    """Walk each entry forward bar-by-bar to hit stop, target, or time.

    Stop is checked before target on a shared bar, the pessimistic assumption
    when intrabar order is unknown.
    """
    high = ny["high"].to_numpy(dtype=float)
    low = ny["low"].to_numpy(dtype=float)
    close = ny["close"].to_numpy(dtype=float)
    spread = ny["spread"].to_numpy(dtype=float) / 100.0
    dates = ny["date"].to_numpy()
    out: list[dict[str, object]] = []
    day_ends: dict[object, int] = {}
    for i in np.flatnonzero(entry):
        d = int(direction[i])
        sl, tp = float(stop[i]), float(target[i])
        risk = abs(close[i] - sl)
        if not np.isfinite(risk) or risk <= 0:
            continue
        reward = d * (tp - close[i])
        if not np.isfinite(tp) or reward <= 0:
            continue  # target already behind price: not a trade
        if dates[i] not in day_ends:
            # Last bar belonging to the same session. bars_left counts bars
            # remaining in the session, so it gives the boundary directly.
            day_ends[dates[i]] = i + int(ny["bars_left"].to_numpy()[i])
        end = min(day_ends[dates[i]], i + MAX_HOLD)
        for j in range(i + 1, end + 1):
            hit_stop = low[j] <= sl if d == 1 else high[j] >= sl
            hit_target = high[j] >= tp if d == 1 else low[j] <= tp
            if hit_stop:
                out.append({"date": dates[i], "r": -1.0, "outcome": "stop", "bars": j - i})
                break
            if hit_target:
                out.append(
                    {
                        "date": dates[i],
                        "r": reward / risk - 2.0 * spread[i] / risk,
                        "outcome": "target",
                        "bars": j - i,
                    }
                )
                break
        else:
            r = d * (close[end] - close[i]) / risk - 2.0 * spread[i] / risk
            out.append({"date": dates[i], "r": r, "outcome": "timeout", "bars": end - i})
    return pd.DataFrame(out)


def summarise(trades: pd.DataFrame, label: str) -> dict[str, object]:
    """Per-session R statistics; the session is the unit of independence."""
    if trades.empty:
        return {"label": label, "n_trades": 0, "n_days": 0}
    per_day = trades.groupby("date")["r"].mean()
    wins = trades[trades["r"] > 0]
    losses = trades[trades["r"] < 0]
    ordered = np.sort(per_day.to_numpy())
    return {
        "label": label,
        "n_trades": len(trades),
        "n_days": len(per_day),
        "total_R": float(trades["r"].sum()),
        "mean_R": float(trades["r"].mean()),
        "median_R": float(trades["r"].median()),
        "win_rate": float((trades["r"] > 0).mean()),
        "profit_factor": (
            float(wins["r"].sum() / abs(losses["r"].sum())) if len(losses) else float("inf")
        ),
        "day_win_rate": float((per_day > 0).mean()),
        # Drop the single best session: a real edge must survive without it.
        "drop_best_day": float(ordered[:-1].mean()) if len(ordered) > 1 else float("nan"),
        "top1_day_R": float(per_day.max()),
        "avg_bars": float(trades["bars"].mean()),
    }


def main() -> int:
    ny = build_ny_frame(load_bars())
    fvg = fvg_zones(ny)
    levels = swing_levels(ny)
    atr = atr_series(ny)
    long_side, _ = sweep_flags(ny)
    disp = displacement_flags(ny)

    ifvg_top = fvg["ifvg_bull_top"]
    ifvg_bot = fvg["ifvg_bull_bot"]
    setup = long_side & disp["disp_long"] & ifvg_top.notna() & ifvg_bot.notna()
    print(f"sweeps long {int(long_side.sum())}, with displacement {int(disp['disp_long'].sum())}")
    print(f"setups (sweep + displacement + open IFVG): {int(setup.sum())}")

    close = ny["close"].to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float)
    buffer = STOP_ATR * np.where(np.isfinite(atr_v), atr_v, np.nan)
    buy_side = levels["buy_side"].to_numpy(dtype=float)

    # Variant A: enter on the displacement bar, no retest confirmation.
    ok = setup.to_numpy() & np.isfinite(buy_side)
    specs: list[
        tuple[
            str,
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
        ]
    ] = [("ifvg_noretest_LONG", ok, np.ones(len(ny), dtype=np.int64), close - buffer, buy_side)]

    # Variant B: wait for price to retest the inverted gap, then enter.
    retest_entry = np.zeros(len(ny), dtype=bool)
    retest_stop = np.full(len(ny), np.nan)
    retest_tgt = np.full(len(ny), np.nan)
    lows_all = ny["low"].to_numpy(dtype=float)
    for i in np.flatnonzero(ok):
        top = float(ifvg_top.iloc[i])
        bot = float(ifvg_bot.iloc[i])
        window = slice(i + 1, min(len(ny), i + 120))
        touched = np.flatnonzero(lows_all[window] <= top)
        if touched.size == 0:
            continue
        j = int(window.start + touched[0])
        retest_entry[j] = True
        retest_stop[j] = bot - STOP_ATR * float(buffer[j])
        retest_tgt[j] = buy_side[j]
    specs.append(
        (
            "ifvg_retest_LONG",
            retest_entry,
            np.ones(len(ny), dtype=np.int64),
            retest_stop,
            retest_tgt,
        )
    )

    # Stop width sweep. The first pass used 0.5 ATR (~7 pts) while the median
    # NY bar range is 12 pts, so the stop sat inside ordinary noise and every
    # trade was noise-stopped. Sweeping finds the width the idea deserves.
    for width in (1.0, 2.0, 3.0, 5.0):
        specs.append(
            (
                f"noretest_stop{int(width)}atr",
                ok,
                np.ones(len(ny), dtype=np.int64),
                close - width * np.where(np.isfinite(atr_v), atr_v, np.nan),
                buy_side,
            )
        )

    frames: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for label, entry, direction, stop, target in specs:
        trades = resolve(ny, entry, direction, stop, target)
        if not trades.empty:
            trades["label"] = label
        frames.append(trades)
        summary.append(summarise(trades, label))
        print(f"\n{label}: {len(trades)} trades")
        if not trades.empty:
            print(trades["outcome"].value_counts().to_string())

    table = pd.DataFrame(summary)
    print("\n=== SUMMARY (per-session R) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "ifvg_chain_summary.csv", index=False)
    live = [f for f in frames if not f.empty]
    if live:
        pd.concat(live, ignore_index=True).to_csv(OUT / "ifvg_chain_trades.csv", index=False)
    print(f"\nwrote {OUT / 'ifvg_chain_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
