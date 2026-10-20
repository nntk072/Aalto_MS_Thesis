"""Does the pre-NY bias carry directional information at all?

Before building ideas 2.1-2.3, test the premise they rest on. Every claim here
is causal and measured against matched controls (same minute-of-NY, same
direction), so the reported number is edge *over drift*, not market direction.

Tested:
  1. Asia sweep-and-reclaim, the ICT liquidity premise, both directions.
  2. Sweep continuation versus reversal (which side the reclaim favours).
  3. Asia-range position (premium/discount) as a directional prior.

Diagnostic only. Writes no training config and launches no run.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from edge_harness import (  # type: ignore[import-not-found,unused-ignore]
    HORIZONS,
    OUT,
    build_ny_frame,
    cash_hold,
    evaluate,
    load_bars,
    show,
    sweep_flags,
    verdict,
)


def main() -> int:
    bars = load_bars()
    ny = build_ny_frame(bars)
    print(f"NY bars: {len(ny)} over {ny['date'].nunique()} sessions")

    hold = cash_hold(ny)
    print("\n=== DRIFT BENCHMARK (hold long all NY session, net pts/day) ===")
    print(hold.groupby("year")["net"].agg(["count", "mean", "median", "std"]).round(2).to_string())

    long_side, short_side = sweep_flags(ny)
    print(f"\nAsia sweeps: long {int(long_side.sum())}  short {int(short_side.sum())}")

    frames: list[pd.DataFrame] = []

    # 1 + 2: sweep and reclaim, in its own direction and reversed.
    sweep_any = long_side | short_side
    frames.append(evaluate(ny, long_side, pd.Series(1, index=ny.index), "asia_sweep_LONG(cont)"))
    frames.append(evaluate(ny, long_side, pd.Series(-1, index=ny.index), "asia_sweep_LONG(rev)"))
    frames.append(evaluate(ny, short_side, pd.Series(-1, index=ny.index), "asia_sweep_SHORT(cont)"))
    frames.append(evaluate(ny, short_side, pd.Series(1, index=ny.index), "asia_sweep_SHORT(rev)"))
    both_dir = pd.Series(np.where(long_side, 1, -1), index=ny.index)
    frames.append(evaluate(ny, sweep_any, both_dir, "asia_sweep_both"))
    # The premise says the reclaim direction wins. Test its mirror explicitly:
    # if continuation is reliably worse than control, fading it is the edge.
    flip_dir = pd.Series(-np.where(long_side, 1, -1), index=ny.index)
    frames.append(evaluate(ny, sweep_any, flip_dir, "asia_sweep_REVERSED"))

    # 3: premium/discount prior, entered at the NY open.
    open_rows = ny["ny_open"].ne(ny["ny_open"].shift(1))
    in_premium = open_rows & (ny["ny_open"] > ny["asia_mid"])
    in_discount = open_rows & (ny["ny_open"] < ny["asia_mid"])
    frames.append(evaluate(ny, in_premium, pd.Series(1, index=ny.index), "open_premium_LONG"))
    frames.append(evaluate(ny, in_discount, pd.Series(1, index=ny.index), "open_discount_LONG"))

    for frame in frames:
        print()
        show(frame[frame["group"].isin(["all"])] if "group" in frame else frame)

    result = pd.concat(frames, ignore_index=True)
    allrows = result[result["group"] == "all"]
    print("\n=== SUMMARY: all groups, horizons " + str(HORIZONS) + " ===")
    show(allrows)

    best = allrows.reindex(allrows["diff"].abs().sort_values(ascending=False).index).head(12)
    print("\n=== 12 largest |diff| (all groups) ===")
    show(best)

    # Chronological gate: first 70% of sessions in time, last 30% held out.
    # Per-year slices above can flatter; this is the honest out-of-sample split.
    uniq_dates = np.array(sorted(ny["date"].unique()))
    split_date = uniq_dates[int(len(uniq_dates) * 0.7)]
    train_only = ny["date"] <= split_date
    print(f"\n=== CHRONOLOGICAL SPLIT (cutoff {split_date}) ===")
    long2, short2 = sweep_flags(ny)
    sweep2 = long2 | short2
    dir2 = pd.Series(np.where(long2, 1, -1), index=ny.index)
    sweep_res = pd.concat(
        [
            evaluate(ny, sweep2 & train_only, dir2, "sweep_cont_TRAIN", by_year=False),
            evaluate(ny, sweep2 & ~train_only, dir2, "sweep_cont_TEST", by_year=False),
            evaluate(ny, sweep2 & train_only, -dir2, "sweep_fade_TRAIN", by_year=False),
            evaluate(ny, sweep2 & ~train_only, -dir2, "sweep_fade_TEST", by_year=False),
        ],
        ignore_index=True,
    )
    show(sweep_res)

    # Tail-hardened verdict over every single-horizon result.
    graded = verdict(result[result["group"] == "all"])
    passing = graded[graded["passes"]]
    print(f"\n=== GATE: {len(passing)}/{len(graded)} results pass all robustness checks ===")
    show(
        graded,
        cols=[
            "label",
            "horizon",
            "n_days_used",
            "diff",
            "trimmed",
            "median_day",
            "win_days",
            "top5_share",
            "drop_best",
            "passes",
            "why",
        ],
    )
    graded.to_csv(OUT / "pre_ny_bias_graded.csv", index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "pre_ny_bias_summary.csv"
    result.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
