from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

ROOT = Path("/scratch/work/nguyenl37/Aalto_MS_Thesis")
CACHE = ROOT / "cache/us100_us500_20241230_20261016"
OUT = ROOT / "outputs/data_refresh_20261017/missing_m1_tick_reconstruction.json"
TZ = "Etc/GMT-3"


def tick_window(path: Path, start: pd.Timestamp) -> pd.DataFrame:
    start_ns = int(start.tz_convert("UTC").value)
    end_ns = start_ns + 60_000_000_000
    return cast(
        pd.DataFrame,
        pq.read_table(
            path,
            columns=["ts_ns", "bid", "ask"],
            filters=[("ts_ns", ">=", start_ns), ("ts_ns", "<", end_ns)],
        ).to_pandas(),
    )


def tick_context(path: Path, start: pd.Timestamp) -> pd.DataFrame:
    start_ns = int(start.tz_convert("UTC").value) - 120_000_000_000
    end_ns = int(start.tz_convert("UTC").value) + 180_000_000_000
    return cast(
        pd.DataFrame,
        pq.read_table(
            path,
            columns=["ts_ns", "bid", "ask"],
            filters=[("ts_ns", ">=", start_ns), ("ts_ns", "<", end_ns)],
        ).to_pandas(),
    )


def main() -> None:
    result: dict[str, object] = {}
    for symbol in ("US100.cash", "US500.cash"):
        bars_path = CACHE / f"{symbol}_M1_v2-full-day.parquet"
        ticks_path = CACHE / f"{symbol}_ticks_v3-validated-full-day.parquet"
        bars = pd.read_parquet(bars_path)
        index = pd.DatetimeIndex(bars.index)
        train = index <= pd.Timestamp("2025-12-31 23:59:59", tz=index.tz)
        ny = bars["session"].astype(str).eq("ny").to_numpy()
        eligible = train & ny
        ny_index = index[eligible]
        gaps: list[pd.Timestamp] = []
        by_day = pd.Series(ny_index, index=ny_index).groupby(ny_index.date)
        for _, day_rows in by_day:
            times = pd.DatetimeIndex(day_rows.to_numpy()).sort_values()
            for left, right in zip(times[:-1], times[1:], strict=True):
                missing = int((right - left).total_seconds() // 60 - 1)
                if missing > 0:
                    gaps.extend(left + pd.Timedelta(minutes=i) for i in range(1, missing + 1))

        gap_rows: list[dict[str, object]] = []
        for timestamp in gaps:
            ticks = tick_window(ticks_path, timestamp)
            context = tick_context(ticks_path, timestamp)
            start_ns = int(timestamp.tz_convert("UTC").value)
            before = context.loc[context["ts_ns"] < start_ns].tail(1)
            after = context.loc[context["ts_ns"] >= start_ns + 60_000_000_000].head(1)
            gap_rows.append(
                {
                    "bar_timestamp": str(timestamp),
                    "tick_count": int(len(ticks)),
                    "last_quote_before_minute": (
                        {
                            "time_utc": str(pd.to_datetime(int(before["ts_ns"].iloc[0]), utc=True)),
                            "bid": float(before["bid"].iloc[0]),
                            "ask": float(before["ask"].iloc[0]),
                        }
                        if len(before)
                        else None
                    ),
                    "first_quote_after_minute": (
                        {
                            "time_utc": str(pd.to_datetime(int(after["ts_ns"].iloc[0]), utc=True)),
                            "bid": float(after["bid"].iloc[0]),
                            "ask": float(after["ask"].iloc[0]),
                        }
                        if len(after)
                        else None
                    ),
                    "bid_ohlc_from_validated_ticks": (
                        {
                            "open": float(ticks["bid"].iloc[0]),
                            "high": float(ticks["bid"].max()),
                            "low": float(ticks["bid"].min()),
                            "close": float(ticks["bid"].iloc[-1]),
                        }
                        if len(ticks)
                        else None
                    ),
                }
            )

        nearby = sorted(
            {
                timestamp
                for gap in gaps
                for timestamp in (gap - pd.Timedelta(minutes=1), gap + pd.Timedelta(minutes=1))
                if timestamp in bars.index
            }
        )
        parity_rows: list[dict[str, object]] = []
        for timestamp in nearby:
            ticks = tick_window(ticks_path, timestamp)
            if not len(ticks):
                parity_rows.append(
                    {
                        "bar_timestamp": str(timestamp),
                        "tick_count": 0,
                        "exact_bid_ohlc_match": False,
                    }
                )
                continue
            expected = {
                "open": float(ticks["bid"].iloc[0]),
                "high": float(ticks["bid"].max()),
                "low": float(ticks["bid"].min()),
                "close": float(ticks["bid"].iloc[-1]),
            }
            actual = {key: float(cast(Any, bars.loc[timestamp, key])) for key in expected}
            parity_rows.append(
                {
                    "bar_timestamp": str(timestamp),
                    "tick_count": int(len(ticks)),
                    "source_m1_ohlc": actual,
                    "bid_tick_ohlc": expected,
                    "bid_ohlc_match_at_source_cent_precision": all(
                        round(actual[key], 2) == round(expected[key], 2) for key in expected
                    ),
                }
            )
        result[symbol] = {
            "missing_train_ny_minute_bars": gap_rows,
            "adjacent_source_bar_tick_parity": parity_rows,
            "all_missing_minutes_have_validated_ticks": all(
                int(cast(Any, row["tick_count"])) > 0 for row in gap_rows
            ),
            "all_adjacent_sample_bars_match_bid_tick_ohlc_at_cent_precision": all(
                row.get("bid_ohlc_match_at_source_cent_precision", False) for row in parity_rows
            ),
        }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
