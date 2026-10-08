"""Render every IFVG chain trade so the failure can be inspected by eye.

Candlesticks, matching ``quant_rl/eval/plots.py``: wick plus open-close body, drawn
per bar so a sweep or a rejection is visible as price action rather than as a
smoothed line. Every liquidity pool in play is drawn, because "swept liquidity" is
a statement about whichever pool the move actually took, not about Asia alone.

The aggregate R totals say the chain loses; they do not say *why*. Each figure
shows the session phase, the pool taken, the IFVG band, entry/stop/target, and the
exit that happened, so a wrong premise shows up on the chart.

Diagnostic only. Writes no training config and launches no run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from edge_harness import (  # type: ignore[import-not-found,unused-ignore]
    NY_TZ,
    OUT,
    atr_series,
    build_ny_frame,
    displacement_flags,
    fvg_zones,
    htf_swing_levels,
    liquidity_pools,
    load_bars,
    session_phase,
    sweep_flags,
    sweep_pools,
    swing_levels,
)
from ifvg_chain_check import STOP_ATR  # type: ignore[import-not-found,unused-ignore]
from matplotlib.patches import Rectangle

FIG_DIR = OUT / "ifvg_chain_plots"
PRE_BARS = 60
POST_BARS = 180
MAX_HOLD = 180

C_SWEEP = "#7e57c2"  # the pool actually taken
C_POOL = "#b0bec5"  # other pools in play, drawn faint
C_IFVG = "#42a5f5"
C_ENTRY = "#26a69a"
C_STOP = "#ef5350"
C_TARGET = "#66bb6a"
C_HTF = "#ffb74d"
C_UP = "#26a69a"
C_DOWN = "#ef5350"

PHASE_BAND = {
    "manipulation": "#fff59d",
    "distribution": "#e1f5fe",
    "late": "#eceff1",
}


def _styles() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cccccc",
            "grid.color": "#eeeeee",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "font.size": 9,
        }
    )


def plot_one(ny: pd.DataFrame, meta: dict[str, Any], out_path: Path) -> None:
    """One trade: context bars, the setup, and what happened next."""
    entry_pos = meta["entry_pos"]
    # The window must never straddle a session boundary. A fixed PRE_BARS reaches
    # back into the previous day's close when the entry fires early in the session,
    # which put "15:30 of the prior day" immediately left of "09:30" on the same
    # axis and made the chart read as a broken timeline. Clamp to the session the
    # entry belongs to, and keep only that date.
    day = ny["date"].to_numpy()
    pos_in_session = ny.groupby("date").cumcount().to_numpy()
    session_start = entry_pos - int(pos_in_session[entry_pos])
    session_len = int((day == day[entry_pos]).sum())
    session_end = min(session_start + session_len - 1, len(ny) - 1)

    lo = max(session_start, entry_pos - PRE_BARS)
    hi = min(session_end + 1, entry_pos + POST_BARS)
    if hi <= entry_pos:
        # A late entry with no room left to the right: show the bars that remain.
        lo = max(session_start, entry_pos - (PRE_BARS + POST_BARS))
        hi = session_end + 1
    win = ny.iloc[lo:hi]
    x = np.arange(len(win))

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Candlesticks, same geometry as quant_rl/eval/plots.py: a high-low wick plus an
    # open-close body, coloured by bar direction.
    body_w = 0.62
    for k, (o, h, low, c) in enumerate(
        zip(win["open"], win["high"], win["low"], win["close"], strict=True)
    ):
        colour = C_UP if c >= o else C_DOWN
        ax.vlines(k, low, h, color=colour, linewidth=0.7, zorder=2)
        body_lo, body_hi = min(o, c), max(o, c)
        if body_hi - body_lo < 1e-9:
            ax.hlines(body_lo, k - body_w / 2, k + body_w / 2, color=colour, linewidth=0.9)
        else:
            ax.add_patch(
                Rectangle(
                    (k - body_w / 2, body_lo),
                    body_w,
                    body_hi - body_lo,
                    facecolor=colour,
                    edgecolor=colour,
                    linewidth=0.5,
                    zorder=3,
                )
            )
    # Every liquidity pool available to this session, so the chart shows which one
    # was actually taken rather than assuming Asia was the relevant one. Off-screen
    # pools are clamped to the frame edge and labelled, because a pool that cannot be
    # seen is indistinguishable from a pool that does not exist.
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    edge_labels: list[str] = []
    for name, level in sorted(meta["pools"].items()):
        taken = name in meta["pool_taken"]
        if not np.isfinite(level):
            continue
        short = name.split("_", 1)[1]
        if level > ymax:
            shown, arrow = ymax - span * 0.012, "above"
        elif level < ymin:
            shown, arrow = ymin + span * 0.012, "below"
        else:
            shown, arrow = level, ""
        ax.axhline(
            shown,
            color=C_SWEEP if taken else C_POOL,
            linewidth=1.6 if taken else 1.0,
            alpha=1.0 if taken else 0.75,
            linestyle="-" if taken else "--",
            zorder=6 if taken else 2,
        )
        if arrow:
            edge_labels.append(f"{short} {arrow}")
        else:
            ax.annotate(
                short,
                xy=(0.995, shown),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=C_SWEEP if taken else "#78909c",
                fontweight="bold" if taken else "normal",
                zorder=7,
            )
    if edge_labels:
        ax.text(
            0.005,
            0.02,
            "off-frame pools: " + ", ".join(edge_labels),
            transform=ax.transAxes,
            fontsize=6.5,
            color="#78909c",
        )
    if meta["pool_taken"]:
        ax.axhline(
            meta["pools"][sorted(meta["pool_taken"])[0]],
            color=C_SWEEP,
            linewidth=1.6,
            zorder=6,
            label=f"SWEPT: {', '.join(sorted(meta['pool_taken']))}",
        )

    # The inverse fair value gap.
    ax.axhspan(
        meta["ifvg_bot"], meta["ifvg_top"], color=C_IFVG, alpha=0.25, zorder=1, label="IFVG zone"
    )
    ax.axhline(meta["ifvg_top"], color=C_IFVG, linewidth=0.8, zorder=3)
    ax.axhline(meta["ifvg_bot"], color=C_IFVG, linewidth=0.8, zorder=3)

    ex = entry_pos - lo
    ax.scatter(
        [ex],
        [meta["entry"]],
        color=C_ENTRY,
        s=70,
        zorder=6,
        marker="o",
        edgecolors="white",
        linewidths=0.8,
        label="entry",
    )
    ax.axhline(
        meta["stop"],
        color=C_STOP,
        linestyle="--",
        linewidth=1.2,
        zorder=4,
        label=f"stop {meta['stop']:.1f}",
    )
    ax.axhline(
        meta["target"],
        color=C_TARGET,
        linestyle=":",
        linewidth=1.3,
        zorder=4,
        label=f"target {meta['target']:.1f}",
    )
    if np.isfinite(meta["htf_buy_side"]):
        ax.axhline(
            meta["htf_buy_side"],
            color=C_HTF,
            linewidth=0.9,
            alpha=0.8,
            zorder=4,
            label=f"HTF buy-side {meta['htf_buy_side']:.1f}",
        )
    if np.isfinite(meta["exit_price"]):
        ax.scatter(
            [meta["exit_pos"] - lo],
            [meta["exit_price"]],
            color=C_STOP,
            s=95,
            zorder=7,
            marker="X",
            label=f"exit: {meta['outcome']}",
        )

    # Session-phase shading, so a late entry is visible at a glance.
    ph = meta["phases"]
    start = 0
    for k in range(1, len(ph) + 1):
        if k == len(ph) or ph[k] != ph[start]:
            if ph[start] in PHASE_BAND:
                ax.axvspan(
                    start - 0.5,
                    k - 0.5,
                    color=PHASE_BAND[ph[start]],
                    alpha=0.25,
                    zorder=0,
                    linewidth=0,
                )
            start = k

    et = pd.DatetimeIndex(win.index).tz_convert(NY_TZ)
    ticks = np.flatnonzero((et.hour * 60 + et.minute) % 30 == 0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([et[i].strftime("%H:%M") for i in ticks], rotation=45, fontsize=7)
    ax.set_xlim(-1, len(win))
    ax.grid(alpha=0.35, zorder=0)
    ax.set_ylabel("US100 price")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.9)

    outcome = meta["outcome"]
    colour = {"stop": "#c62828", "target": "#2e7d32"}.get(outcome, "#ef6c00")
    ax.set_title(
        f"{pd.Timestamp(meta['date']).date()}   entry {meta['entry_et']} ET   "
        f"phase={meta['phase']}   outcome={outcome}   R={meta['r']:+.2f}   "
        f"held {meta['bars']} bars",
        color=colour,
        fontsize=11,
        fontweight="bold",
    )

    risk = meta["entry"] - meta["stop"]
    reward = meta["target"] - meta["entry"]
    rr = reward / risk if risk > 0 else np.nan
    ax.text(
        0.01,
        -0.14,
        f"risk {risk:.1f} pts | reward {reward:.1f} pts | RR {rr:.1f}:1 | "
        f"ATR {meta['atr']:.1f} | stop = {STOP_ATR}x ATR | "
        f"pools swept: {', '.join(sorted(meta['pool_taken'])) or 'none flagged'}",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )

    axr.bar(x, win["spread"].to_numpy() / 100.0, color="#90a4ae", width=1.0)
    axr.set_ylabel("spread\n(pts)", fontsize=8)
    axr.grid(alpha=0.6)
    axr.set_xlabel("New York time (ET)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    _styles()
    bars = load_bars()
    ny = build_ny_frame(bars)
    htf = htf_swing_levels(bars, ny)
    pools = liquidity_pools(bars, ny)
    phase = session_phase(ny)
    sweep_flags_map = sweep_pools(ny, pools)
    fvg = fvg_zones(ny)
    disp = displacement_flags(ny)
    long_side, _ = sweep_flags(ny)
    atr = atr_series(ny)

    ifvg_top = fvg["ifvg_bull_top"]
    ifvg_bot = fvg["ifvg_bull_bot"]
    setup = long_side & disp["disp_long"] & ifvg_top.notna() & ifvg_bot.notna()

    close = ny["close"].to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float)
    buffer = STOP_ATR * np.where(np.isfinite(atr_v), atr_v, np.nan)
    buy_side = swing_levels(ny)["buy_side"].to_numpy(dtype=float)
    ok = setup.to_numpy() & np.isfinite(buffer) & np.isfinite(buy_side)

    setup_pos = np.flatnonzero(ok)
    print(f"{len(setup_pos)} setups to plot")

    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()

    low_all = ny["low"].to_numpy(dtype=float)
    high_all = ny["high"].to_numpy(dtype=float)
    close_all = ny["close"].to_numpy(dtype=float)
    bars_left = ny["bars_left"].to_numpy()

    rows: list[dict[str, Any]] = []
    for k, i in enumerate(setup_pos):
        sl = close[i] - buffer[i]
        tp = buy_side[i]
        # Never walk past the session boundary, matching the research run.
        end = int(min(len(ny) - 1, i + MAX_HOLD, i + bars_left[i]))
        exit_price, exit_pos, outcome, bars_held = np.nan, i, "timeout", MAX_HOLD
        for j in range(i + 1, end + 1):
            if low_all[j] <= sl:
                exit_price, exit_pos, outcome, bars_held = sl, j, "stop", j - i
                break
            if high_all[j] >= tp:
                exit_price, exit_pos, outcome, bars_held = tp, j, "target", j - i
                break
        else:
            exit_price = float(close_all[end])
            bars_held = end - i

        risk = abs(close[i] - sl)
        if outcome == "stop":
            r = -1.0
        elif risk > 0:
            r = (exit_price - close[i]) / risk
        else:
            r = np.nan

        pool_row = pools.loc[ny.index[i]]
        sweep_row = sweep_flags_map
        # Mirror plot_one's clamping so the phase shading covers exactly the bars
        # that get drawn; otherwise the bands drift out of register with the axis.
        _pos = ny.groupby("date").cumcount().to_numpy()
        _days = ny["date"].to_numpy()
        _sess_start = i - int(_pos[i])
        _sess_len = int((_days == _days[i]).sum())
        _sess_end = min(_sess_start + _sess_len - 1, len(ny) - 1)
        lo_i = max(_sess_start, i - PRE_BARS)
        hi_i = min(_sess_end + 1, i + POST_BARS)
        if hi_i <= i:
            lo_i = max(_sess_start, i - (PRE_BARS + POST_BARS))
            hi_i = _sess_end + 1
        meta = {
            "date": ny["date"].iloc[i],
            "entry_pos": int(i),
            "entry": float(close[i]),
            "stop": float(sl),
            "target": float(tp),
            "atr": float(atr_v[i]),
            "ifvg_top": float(ifvg_top.iloc[i]),
            "ifvg_bot": float(ifvg_bot.iloc[i]),
            "htf_buy_side": float(htf["buy_side_htf"].iloc[i]),
            "outcome": outcome,
            "r": float(r),
            "bars": int(bars_held),
            "exit_price": float(exit_price),
            "exit_pos": int(exit_pos),
            "entry_et": pd.Timestamp(ny.index[i]).tz_convert(NY_TZ).strftime("%H:%M"),
            "phase": str(phase["phase"].iloc[i]),
            "phases": [str(v) for v in phase["phase"].iloc[lo_i:hi_i]],
            "pools": {
                col: float(pool_row[col])
                for col in pool_row.index
                if col.startswith(("bs_", "ss_")) and pd.notna(pool_row[col])
            },
            "pool_taken": [name for name, f in sweep_row.items() if bool(f.iloc[i])],
        }
        name = f"trade_{k:03d}_{pd.Timestamp(ny['date'].iloc[i]).date()}_{outcome}.png"
        plot_one(ny, meta, fig_dir / name)
        rows.append(
            {
                "file": name,
                "date": ny["date"].iloc[i],
                "entry_et": meta["entry_et"],
                "phase": meta["phase"],
                "pools_swept": "|".join(sorted(meta["pool_taken"])),
                "outcome": outcome,
                "r": float(r),
                "bars": int(bars_held),
                "risk_pts": round(float(risk), 2),
                "reward_pts": round(float(tp - close[i]), 2),
            }
        )

    index = pd.DataFrame(rows)
    index.to_csv(fig_dir / "index.csv", index=False)
    print(f"wrote {len(index)} figures + index.csv to {fig_dir}")
    if not index.empty:
        print(index["outcome"].value_counts().to_string())
        print(f"total R: {index['r'].sum():+.2f}")
        print(
            f"median risk {index['risk_pts'].median():.1f} pts, "
            f"median reward {index['reward_pts'].median():.1f} pts"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
