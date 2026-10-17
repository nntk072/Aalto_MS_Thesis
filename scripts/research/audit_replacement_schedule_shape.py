from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path("/scratch/work/nguyenl37/Aalto_MS_Thesis")
CACHE = ROOT / "cache/us100_us500_20241230_20261016"
OUT = ROOT / "outputs/data_refresh_20261017/replacement_schedule_shape.json"


def main() -> None:
    report: dict[str, object] = {}
    for symbol in ("US100.cash", "US500.cash"):
        path = CACHE / f"{symbol}_M1_v2-full-day.parquet"
        bars = pd.read_parquet(path, columns=["session"])
        index = pd.DatetimeIndex(bars.index)
        train = index <= pd.Timestamp("2025-12-31 23:59:59", tz=index.tz)
        eligible = bars["session"].astype(str).eq("ny")
        daily = cast(
            pd.Series,
            eligible.loc[train].groupby(index[train].date).sum().astype("int64"),
        )
        observed_dates = daily.loc[daily.to_numpy(dtype=np.int64) > 0]
        intraday_gaps: list[dict[str, object]] = []
        ny_rows = bars.loc[train & eligible.to_numpy()]
        ny_index = pd.DatetimeIndex(ny_rows.index)
        ny_dates = pd.Series(ny_index.date, index=ny_index)
        for day, day_index in ny_rows.groupby(ny_dates).groups.items():
            times = pd.DatetimeIndex(day_index).sort_values()
            deltas = times.to_series().diff().dt.total_seconds().to_numpy(dtype=np.float64)
            for position in np.flatnonzero(deltas > 60):
                previous = times[int(position) - 1]
                current = times[int(position)]
                gap = int((current - previous).total_seconds() // 60 - 1)
                intraday_gaps.append(
                    {
                        "date": str(day),
                        "previous_bar": str(previous),
                        "next_bar": str(current),
                        "gap_minutes_between_observed_rows": gap,
                    }
                )
        report[symbol] = {
            "path": str(path),
            "gap_semantics": (
                "Counts elapsed minute labels between observed NY-session rows; "
                "does not establish expected trading hours or data loss."
            ),
            "train_ny_rows": int(daily.sum()),
            "train_dates_with_ny_rows": int(len(observed_dates)),
            "intra_ny_gaps_over_one_minute": intraday_gaps,
            "row_counts_per_observed_date": {
                str(k): int(v) for k, v in observed_dates.value_counts().sort_index().items()
            },
            "dates_with_fewer_than_391_ny_rows": {
                str(k): int(v) for k, v in observed_dates[observed_dates < 391].items()
            },
            "dates_with_more_than_391_ny_rows": {
                str(k): int(v) for k, v in observed_dates[observed_dates > 391].items()
            },
        }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
