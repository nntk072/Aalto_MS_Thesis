"""Counterfactual replay: would a fallback take-profit have helped the EMA-21 exit?

Reads a completed run's ``trades.csv`` and walks the raw M1 bars between each
EMA-21 open and its logged close. For a hypothetical fixed-R target it answers
one question: would the price have reached that target before the trade ended?

This is diagnostic only. It changes no training code and launches no run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

RUN = Path("outputs/20261003_171903_512479_rl_train_seed50_tcn/training")
BARS = Path("cache/US100.cash_M1_v2-full-day.parquet")


CAPS: tuple[float, ...] = (8.0, 6.0, 4.0, 3.0, 2.0)
CONTROL_SEED = 20261003


def load_pairs(run: Path = RUN) -> list[dict[str, Any]]:
    """Walk the log in order, pairing each open with the close that follows it.

    Rows are strictly interleaved ``open`` then its single ``close``-family row,
    so walking forward past ``reject`` rows keeps every trade correctly paired.
    Positionally zipping two filtered frames mispairs the tail.
    """
    trades = pd.read_csv(run / "trades.csv")
    close_types = ("close", "stop_close", "tp_close", "eod_close", "forced_close")
    pairs: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for row in trades.to_dict("records"):
        record = cast("dict[str, Any]", row)
        kind = str(record["type"])
        if kind == "open":
            pending = dict(record)
            continue
        if pending is None:
            continue
        if kind in close_types:
            pending["close_time"] = record["time"]
            pending["close_reason"] = record["reason"]
            pending["close_price"] = record["price"]
            pairs.append(pending)
            pending = None
    return pairs


def replay(pairs: list[dict[str, Any]], bars: pd.DataFrame) -> pd.DataFrame:
    """For each EMA trade, find the best R multiple the price reached."""
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    times = bars.index
    # True range on the raw bars; the trade log does not record atr_5.
    prev_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    out: list[dict[str, Any]] = []
    for row in pairs:
        if str(row["exit_mode"]) != "ema_21":
            continue
        t_open = pd.Timestamp(row["time"])
        t_close = pd.Timestamp(row["close_time"])
        if pd.isna(t_open) or pd.isna(t_close):
            continue
        lo = int(times.searchsorted(t_open))
        hi = int(times.searchsorted(t_close, side="right"))
        if hi <= lo:
            continue
        entry = float(row["price"])
        risk = float(row["stop_distance"])
        if risk <= 0:
            continue
        # Best favorable excursion in R over the trade's life.
        if int(row["direction"]) == 1:
            mfe = (float(high[lo:hi].max()) - entry) / risk
        else:
            mfe = (entry - float(low[lo:hi].min())) / risk
        atr = float(np.median(true_range[max(lo - 14, 0) : lo + 1]))
        out.append(
            {
                "realized_r": float(row["realized_r"]),
                "mfe_r": mfe,
                "planned_rr": row["planned_rr"],
                "direction": int(row["direction"]),
                "reason": row["close_reason"],
                "stop_distance": risk,
                "sl_family": sl_family(row["sl_ref"]),
                "atr_5": atr,
                # Ratio the cap test keys on: a structural stop wider than
                # ``cap`` x ATR is refused outright, not clamped.
                "stop_atr_mult": (risk / atr) if atr > 0 else float("nan"),
                "volatility_regime": row["volatility_regime"],
                "trend_regime": row["trend_regime"],
                "session": row["session"],
                "manipulation": row["manipulation"],
            }
        )
    return pd.DataFrame(out)


def report(df: pd.DataFrame, targets: tuple[float, ...]) -> None:
    """Print baseline versus counterfactual expectancy at several targets."""
    n = len(df)
    base = df["realized_r"].sum()
    print(f"EMA-21 trades analysed: {n}")
    print(f"baseline total R = {base:+.2f}   per trade = {base / n:+.5f}")
    print(f"share that reached +1R = {(df['mfe_r'] >= 1.0).mean():.1%}")
    print(f"median MFE = {df['mfe_r'].median():.2f}R")
    print()
    print("--- why EMA trades end (logged close reason) ---")
    print(df["reason"].value_counts().to_string())
    print()
    print("--- MFE distribution ---")
    print(f"  never left stop territory (MFE < 0.25R): {(df['mfe_r'] < 0.25).mean():.1%}")
    print(
        f"  MFE 0.25-1R (died before the EMA trigger): "
        f"{((df['mfe_r'] >= 0.25) & (df['mfe_r'] < 1.0)).mean():.1%}"
    )
    print(f"  MFE >= 1R (EMA trigger reachable):        {(df['mfe_r'] >= 1.0).mean():.1%}")
    print()


def _bucket_table(df: pd.DataFrame, targets: tuple[float, ...]) -> None:
    """Realised outcome and profit given back, bucketed by MFE."""
    print("--- realised outcome conditioned on how far the trade went ---")
    buckets = [
        ("MFE < 0.25R", df["mfe_r"] < 0.25),
        ("0.25-1R", (df["mfe_r"] >= 0.25) & (df["mfe_r"] < 1.0)),
        ("1-1.5R", (df["mfe_r"] >= 1.0) & (df["mfe_r"] < 1.5)),
        ("MFE >= 1.5R", df["mfe_r"] >= 1.5),
    ]
    n = len(df)
    print(
        f"{'bucket':>12} {'n':>6} {'share':>7} {'mean real':>10} {'total R':>10} {'given back':>11}"
    )
    for label, mask in buckets:
        sub = df.loc[mask]
        if sub.empty:
            continue
        given = sub["mfe_r"].mean() - sub["realized_r"].mean()
        print(
            f"{label:>12} {len(sub):>6} {len(sub) / n:>6.1%} "
            f"{sub['realized_r'].mean():>+10.3f} {sub['realized_r'].sum():>+10.2f} {given:>+11.3f}"
        )
    print()
    print("--- oracle bound: entry quality, ignoring exit rule ---")
    print(f"  total R if every trade exited at its own MFE = {df['mfe_r'].sum():+.2f}")
    print(f"  per trade                                     = {df['mfe_r'].sum() / n:+.5f}")
    print()
    print(f"{'target':>8} {'hit%':>7} {'total R':>10} {'per trade':>11} {'delta R':>10}")
    base = df["realized_r"].sum()
    for k in targets:
        hit = df["mfe_r"] >= k
        # A trade that reaches the target pays kR; otherwise it keeps its
        # logged outcome. Breakeven is already active at +1R in this run.
        total = float((df["realized_r"].where(~hit, k)).sum())
        print(
            f"{k:>7.1f}R {hit.mean():>6.1%} {total:>+10.2f} "
            f"{total / n:>+11.5f} {total - base:>+10.2f}"
        )


def sl_family(ref: str) -> str:
    """Coarse family for a stop reference, mirroring the run's own grouping."""
    text = str(ref)
    for family in ("po3_manipulation", "london", "asian", "fvg", "ifvg", "swing", "dev_"):
        if family in text:
            return family
    return "other"


def _slice_table(df: pd.DataFrame, key: str, label: str) -> None:
    """Print reachability and expectancy grouped by one categorical column."""
    print(f"--- MFE reachability and expectancy by {label} ---")
    print(
        f"{label:>16} {'n':>5} {'med stop':>9} {'med MFE':>8} "
        f"{'reached 1R':>11} {'mean real':>10} {'total R':>10}"
    )
    for value, sub in df.groupby(key, dropna=False, observed=True):
        if len(sub) < 5:
            continue
        med_stop = float(sub["stop_distance"].median()) if "stop_distance" in sub else float("nan")
        print(
            f"{str(value)[:16]:>16} {len(sub):>5} {med_stop:>9.2f} "
            f"{sub['mfe_r'].median():>8.2f} {(sub['mfe_r'] >= 1.0).mean():>10.1%} "
            f"{sub['realized_r'].mean():>+10.3f} {sub['realized_r'].sum():>+10.2f}"
        )
    print()


def _stop_width_table(df: pd.DataFrame) -> None:
    """Group by stop width tercile: is 1R simply out of reach for wide stops?"""
    cuts = df["stop_distance"].quantile([0.0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    widths = df["stop_distance"]
    buckets = pd.cut(
        widths,
        bins=[float(cuts[0]), float(cuts[1]), float(cuts[2]), float(cuts[3])],
        labels=["tight (low 3rd)", "mid", "wide (top 3rd)"],
        include_lowest=True,
    )
    df = df.assign(width=buckets)
    _slice_table(df, "width", "stop width tercile")


def _atr_ratio_table(df: pd.DataFrame) -> None:
    """Group by stop distance in ATR terms, coarse enough to stay readable."""
    ratio = (df["stop_distance"] / df["atr_5"]).round(1)
    df = df.assign(
        atr_mult=pd.cut(
            ratio,
            bins=[0.0, 1.0, 2.0, 4.0, 8.0, float("inf")],
            labels=["<1x", "1-2x", "2-4x", "4-8x", ">8x"],
        )
    )
    _slice_table(df, "atr_mult", "stop distance (x ATR)")


def entry_selection_report(df: pd.DataFrame) -> None:
    """Ask whether the trades that ran can be picked out at entry."""
    print("=" * 72)
    print("ENTRY-SELECTION QUESTION: can we identify the runners up front?")
    print("=" * 72)
    print()
    _slice_table(df, "sl_family", "SL family")
    _stop_width_table(df)
    _atr_ratio_table(df)
    _slice_table(df, "volatility_regime", "volatility regime")
    _slice_table(df, "trend_regime", "trend regime")

    print("--- cumulative expectancy by 'did it reach 1R' (selection check) ---")
    ran = df.loc[df["mfe_r"] >= 1.0]
    died = df.loc[df["mfe_r"] < 1.0]
    print(
        f"  reached 1R: n={len(ran):>4} mean={ran['realized_r'].mean():+.3f} "
        f"total={ran['realized_r'].sum():+.2f}"
    )
    print(
        f"  never 1R : n={len(died):>4} mean={died['realized_r'].mean():+.3f} "
        f"total={died['realized_r'].sum():+.2f}"
    )
    print("  -> if 'reached 1R' were known at entry, expectancy would be positive.")
    print("     It is not knowable in advance, so this bounds, not proves, an edge.")
    print()
    print("--- best single feature split on 1R reachability ---")
    for key in ("sl_family", "volatility_regime", "trend_regime", "session", "manipulation"):
        if key not in df.columns:
            continue
        grouped = df.groupby(key, dropna=False)
        sizes = grouped.size()
        reach = df["mfe_r"] >= 1.0
        rates = reach.groupby([df[key], df.index]).mean().groupby(level=0).mean()
        rates = rates[sizes.reindex(rates.index) >= 20]
        if rates.empty:
            continue
        print(
            f"  {key:<20} spread = {rates.max() - rates.min():.1%}  "
            f"(best={rates.idxmax()} {rates.max():.1%} / "
            f"worst={rates.idxmin()} {rates.min():.1%})"
        )


def cap_sweep(df: pd.DataFrame, caps: tuple[float, ...] = CAPS) -> None:
    """Would refusing over-wide stops have helped, and is the gain real?

    Refusing trades raises mean R almost by construction: drop enough losers and
    the average rises even when nothing was selected intelligently. So each cap
    is also compared against random subsets of the *same size* drawn from the
    full trade list. A cap that merely deletes trades lands on the control; a
    cap that keeps genuinely better trades beats it.
    """
    print("=" * 72)
    print("STOP-WIDTH CAP SWEEP: refuse the entry when stop > cap x ATR")
    print("=" * 72)
    n = len(df)
    r = df["realized_r"].to_numpy(dtype=float)
    base_mean = float(r.mean())
    base_total = float(r.sum())
    print(f"baseline: n={n} total={base_total:+.2f}R mean={base_mean:+.4f}R/trade")
    print()
    rng = np.random.default_rng(CONTROL_SEED)
    print(
        f"{'cap':>6} {'kept':>6} {'ret%':>7} {'total R':>9} {'mean R':>9} "
        f"{'1R hit':>7} {'ctrl mean':>10} {'edge':>8}"
    )
    for cap in caps:
        kept = df.loc[df["stop_atr_mult"] <= cap]
        if kept.empty:
            print(f"{cap:>5.1f}x {0:>6} {'-':>7} {'-':>9} {'-':>9} {'-':>7} {'-':>10} {'-':>8}")
            continue
        k = len(kept)
        km = float(kept["realized_r"].mean())
        # Control: same number of trades, chosen at random instead of by ATR.
        ctrl = float(rng.choice(r, size=k, replace=False).mean()) if k < n else base_mean
        edge = km - ctrl
        print(
            f"{cap:>5.1f}x {k:>6} {k / n:>6.1%} {kept['realized_r'].sum():>+9.2f} "
            f"{km:>+9.4f} {(kept['mfe_r'] >= 1.0).mean():>6.1%} {ctrl:>+10.4f} {edge:>+8.4f}"
        )
    print()
    print("'edge' is kept-mean minus same-size random mean. Read it as: does the cap")
    print("select better trades, or just fewer trades? Retention below ~60% means the")
    print("cap is deleting the strategy rather than improving it.")
    print()


def atr_exit_sweep(
    df: pd.DataFrame,
    targets: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
) -> None:
    """Would exiting at a fixed ATR multiple beat exiting at 1R?

    The stop-width cap failed because it could only delete trades. This tests
    the other lever: keep every trade, but trigger the exit at ``k`` x ATR of
    favorable movement instead of at 1R of stop distance. A wide stop currently
    pushes the 1R trigger out of reach; an ATR trigger does not care how wide
    the stop is.

    Exit at ``k`` x ATR pays ``k / stop_atr_mult`` R, so on a wide stop it pays
    well under 1R, while on a tight stop it pays over 1R. Trades that never
    reach the target keep their logged outcome.
    """
    print("=" * 72)
    print("ATR-BASED EXIT SWEEP: exit at k x ATR instead of 1R of stop distance")
    print("=" * 72)
    n = len(df)
    r = df["realized_r"].to_numpy(dtype=float)
    base_total = float(r.sum())
    print(f"baseline (1R trigger): n={n} total={base_total:+.2f}R mean={base_total / n:+.4f}")
    print()
    print(
        f"{'k':>6} {'paid R':>7} {'changed':>8} {'ret%':>7} {'total R':>9} "
        f"{'mean R':>9} {'vs base':>9}"
    )
    for k in targets:
        paid = k / df["stop_atr_mult"].replace(0.0, np.nan)
        fires = df["mfe_r"].to_numpy() >= paid.to_numpy()
        cf = np.where(fires, paid.to_numpy(), r)
        changed = int(fires.sum())
        print(
            f"{k:>5.2f}x {float(paid.mean()):>7.3f} {changed:>8} {changed / n:>6.1%} "
            f"{cf.sum():>+9.2f} {cf.mean():>+9.4f} {cf.sum() - base_total:>+9.2f}"
        )
    print()
    print("'paid R' is the average payoff per triggered trade in R; it falls as the")
    print("stop widens, which is the trade-off against firing on many more trades.")
    print("'vs base' is the change in total R versus the logged 1R trigger. Every row")
    print("keeps all n trades, so total R is directly comparable and not a mean-of-")
    print("survivors artifact the way the stop-width cap was.")
    print()


def cost_check(df: pd.DataFrame, k: float = 0.5) -> None:
    """Is the ATR-exit profit bigger than the cost of taking it?

    Exiting at a fraction of an ATR pays a fraction of an R, which in points
    can be smaller than the spread plus commission. If so the sweep's positive
    total R is an artifact of a frictionless replay.
    """
    print("=" * 72)
    print(f"COST CHECK at k={k}x ATR: profit in points vs round-turn friction")
    print("=" * 72)
    sub = df.loc[df["stop_atr_mult"] > 0]
    target_pts = k * sub["atr_5"]
    print(
        f"  target distance: median {float(target_pts.median()):.2f} pts, "
        f"p10 {float(target_pts.quantile(0.10)):.2f}, "
        f"p90 {float(target_pts.quantile(0.90)):.2f}"
    )
    for spread in (0.5, 1.0, 2.0, 3.0):
        # Round trip pays the spread on entry and exit, plus commission.
        friction = 2 * spread + 0.5
        med = float(target_pts.median()) - friction
        share = float((target_pts > friction).mean())
        print(
            f"  spread {spread:.1f} pts -> friction {friction:.2f} pts: "
            f"net median {med:+.2f} pts, {share:.1%} of trades still profitable"
        )
    print()
    print("If the profitable share collapses as the spread rises, the ATR exit is")
    print("a spread-eating scalp and the sweep's R numbers do not survive.")
    print()


def _decompose(df: pd.DataFrame, k: float = 0.5) -> None:
    """Where does the ATR-exit gain actually come from?

    If most of the improvement is trades that were going to be stopped out, the
    'edge' is just cutting losers early and the strategy is unchanged. If the
    gain came from trades that logged a real win, it is something else.
    """
    print("=" * 72)
    print(f"DECOMPOSITION at k={k}x ATR: which logged trades did we intercept?")
    print("=" * 72)
    paid = k / df["stop_atr_mult"].replace(0.0, np.nan)
    fires = df["mfe_r"].to_numpy() >= paid.to_numpy()
    r = df["realized_r"].to_numpy(dtype=float)
    pf = paid.to_numpy()
    print(f"{'logged outcome':>18} {'n':>6} {'base R':>9} {'cf R':>9} {'delta':>9}")
    groups = [
        ("losers (<0R)", r < 0.0),
        ("flat (0R)", r == 0.0),
        ("winners (>0R)", r > 0.0),
    ]
    for label, mask in groups:
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            continue
        print(
            f"{label:>18} {int(m.sum()):>6} {r[m].sum():>+9.2f} "
            f"{np.where(fires[m], pf[m], r[m]).sum():>+9.2f} "
            f"{np.where(fires[m], pf[m], r[m]).sum() - r[m].sum():>+9.2f}"
        )
    print()
    # Split the intercepted losers by whether the trade ended at the stop.
    # The log's stop-out label is ``structure_sl``; ``stop_close`` never appears.
    losers = np.asarray(r < 0.0, dtype=bool)
    stopped = losers & df["reason"].astype(str).isin(("structure_sl", "stop_close")).to_numpy()
    print(
        f"  of {int(losers.sum())} logged losers, {int(stopped.sum())} ended at the stop "
        f"({stopped.sum() / max(losers.sum(), 1):.1%})"
    )
    print(
        f"  intercept rate on stopped trades: "
        f"{float(fires[stopped].mean()) if stopped.any() else 0.0:.1%}"
    )
    print()
    print("A high intercept rate on stopped trades means the sweep is mostly")
    print("'give up early instead of taking the loss', not 'find the exit earlier'.")
    print()


def main() -> int:
    run = RUN
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    print(f"run: {run}")
    bars = pd.read_parquet(BARS)
    pairs = load_pairs(run)
    df = replay(pairs, bars)
    if df.empty:
        print("no replayable EMA-21 trades found")
        return 1
    report(df, (1.0, 1.5, 2.0, 2.5, 3.0))
    _bucket_table(df, (1.0, 1.5, 2.0, 2.5, 3.0))
    entry_selection_report(df)
    cap_sweep(df)
    atr_exit_sweep(df)
    cost_check(df)
    _decompose(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
