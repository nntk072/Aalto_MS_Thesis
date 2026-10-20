"""Is displacement the leg the sweep test could not isolate?

The pre-NY bias check showed sweep-and-reclaim continuation is reliably *negative*, which is
what a failed take looks like rather than a signal. The untested ICT primitive
is impulsive displacement away from the level: a reclaim only matters if the
move behind it is real. This tests that, and its fade, under the tail gate.

Every result is judged by :func:`edge_harness.verdict`, so a candidate cannot
pass on a handful of lucky sessions.

Diagnostic only. Writes no training config and launches no run.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from edge_harness import (  # type: ignore[import-not-found,unused-ignore]
    OUT,
    build_ny_frame,
    displacement_flags,
    evaluate,
    load_bars,
    show,
    sweep_flags,
    verdict,
)

GRADE_COLS = [
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
]


def main() -> int:
    ny = build_ny_frame(load_bars())
    disp = displacement_flags(ny)
    long_side, short_side = sweep_flags(ny)

    sweep_any = long_side | short_side
    disp_long = disp["disp_long"]
    disp_short = disp["disp_short"]
    disp_any = disp_long | disp_short
    print(f"NY bars {len(ny)} over {ny['date'].nunique()} sessions")
    print(f"displacement events: long {int(disp_long.sum())}  short {int(disp_short.sum())}")

    # Displacement should reduce the raw sweep count if it is a real filter.
    print(f"sweeps {int(sweep_any.sum())} -> with displacement {int(disp_any.sum())}")

    sweep_dir = pd.Series(np.where(long_side, 1, -1), index=ny.index)
    disp_dir = pd.Series(np.where(disp_long, 1, -1), index=ny.index)

    frames = [
        evaluate(ny, disp_any, disp_dir, "disp_CONT"),
        evaluate(ny, disp_any, -disp_dir, "disp_FADE"),
        evaluate(ny, disp_long, pd.Series(1, index=ny.index), "disp_long_LONG"),
        evaluate(ny, disp_short, pd.Series(-1, index=ny.index), "disp_short_SHORT"),
        # Sweep events that displacement did NOT confirm, as a contrast group.
        evaluate(ny, sweep_any & ~disp_any, sweep_dir, "sweep_no_disp_CONT"),
    ]
    result = pd.concat(frames, ignore_index=True)
    allrows = result[result["group"] == "all"]

    graded = verdict(allrows)
    print(f"\n=== GATE: {int(graded['passes'].sum())}/{len(graded)} pass ===")
    show(graded, cols=GRADE_COLS)

    # Honest OOS check on whatever survives.
    uniq_dates = np.array(sorted(ny["date"].unique()))
    split_date = uniq_dates[int(len(uniq_dates) * 0.7)]
    train_only = ny["date"] <= split_date
    surviving = sorted(graded[graded["passes"]]["label"].unique())
    print(f"\n=== CHRONOLOGICAL SPLIT (cutoff {split_date}) ===")
    if not surviving:
        print("No candidate passed the gate; no split to validate.")
    else:
        mapping: dict[str, tuple[pd.Series, pd.Series]] = {
            "disp_CONT": (disp_any, disp_dir),
            "disp_FADE": (disp_any, -disp_dir),
            "disp_long_LONG": (disp_long, pd.Series(1, index=ny.index)),
            "disp_short_SHORT": (disp_short, pd.Series(-1, index=ny.index)),
            "sweep_no_disp_CONT": (sweep_any & ~disp_any, sweep_dir),
        }
        split_rows = []
        for label in surviving:
            mask, direction = mapping[label]
            split_rows.append(
                evaluate(ny, mask & train_only, direction, f"{label}_TRAIN", by_year=False)
            )
            split_rows.append(
                evaluate(ny, mask & ~train_only, direction, f"{label}_TEST", by_year=False)
            )
        show(verdict(pd.concat(split_rows, ignore_index=True)), cols=GRADE_COLS)

    OUT.mkdir(parents=True, exist_ok=True)
    graded.to_csv(OUT / "sweep_displacement_graded.csv", index=False)
    result.to_csv(OUT / "sweep_displacement_summary.csv", index=False)
    print(f"\nwrote {OUT / 'sweep_displacement_graded.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
