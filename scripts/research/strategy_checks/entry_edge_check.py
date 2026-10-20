"""Does the EMA-21 entry signal have an edge at all, independent of any exit?

Every exit-side experiment so far has failed. This asks the prior question:
are the entries themselves profitable, before any stop or target exists?

For each logged entry it measures the forward return over several horizons,
net of the *actual* spread from the bar data (recorded in 1/100 pt), and
compares it against matched controls:

  * same-minute-of-session random entries, and
  * all entries regardless of signal.

The controls absorb market drift and time-of-day effects. Without them, a
positive number could just mean US100 drifted up over the sample period.

This is diagnostic only. It changes no training code and launches no run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

BARS = Path("cache/US100.cash_M1_v2-full-day.parquet")
RUNS = (
    Path("outputs/20261003_171903_512479_rl_train_seed50_tcn"),
    Path("outputs/20261003_171903_426375_rl_train_seed50_tcn"),
)
HORIZONS = (5, 15, 30, 60, 120)
SEED = 20261003


def load_entries(run: Path, split: str) -> pd.DataFrame:
    """Open records with their direction and the session minute they fired in."""
    trades = pd.read_csv(run / split / "trades.csv")
    opens = trades.loc[trades["type"].astype(str).eq("open")].copy()
    if opens.empty:
        return opens
    opens["t"] = pd.to_datetime(opens["time"], utc=True).dt.tz_convert("Etc/GMT-3")
    opens["minute_of_session"] = opens["t"].dt.hour * 60 + opens["t"].dt.minute
    return opens


def forward_returns(
    opens: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Net forward return in points per direction, after the real spread.

    Uses close-to-close so intrabar path is not credited, and charges the
    round trip at the spread recorded on the entry bar.
    """
    close = bars["close"].to_numpy()
    spread_pts = bars["spread"].to_numpy() / 100.0
    times = cast("pd.DatetimeIndex", bars.index)
    stamps = pd.to_datetime(opens["time"], utc=True).dt.tz_convert("Etc/GMT-3")
    minutes = (stamps.dt.hour * 60 + stamps.dt.minute).to_numpy(dtype=np.int64)
    directions = opens["direction"].to_numpy(dtype=np.int64)
    out: list[dict[str, float]] = []
    for k in range(len(opens)):
        i = int(times.searchsorted(stamps.iloc[k]))
        j = i + horizon
        if i <= 0 or j >= len(close):
            continue
        d = int(directions[k])
        raw = float(close[j] - close[i]) * d
        out.append(
            {
                "net": raw - 2.0 * float(spread_pts[i]),
                "raw": raw,
                "minute_of_session": float(minutes[k]),
                "direction": float(d),
            }
        )
    return pd.DataFrame(out)


def _stats(values: np.ndarray[Any, Any]) -> str:
    """Mean with a standard error, so a flat result is visibly flat."""
    n = len(values)
    if n == 0:
        return "     n=0"
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return f"{mean:>+8.2f} +/- {se:4.2f}"


def report(opens: pd.DataFrame, bars: pd.DataFrame, label: str) -> None:
    """Entries versus two controls across horizons."""
    print("=" * 74)
    print(f"ENTRY EDGE: {label}")
    print("=" * 74)
    print(f"entries: {len(opens)}")
    print()
    rng = np.random.default_rng(SEED)
    close = bars["close"].to_numpy()
    spread_pts = bars["spread"].to_numpy() / 100.0
    times = cast("pd.DatetimeIndex", bars.index)
    # Index bars by minute-of-session so controls match the session, not the clock.
    minute_of_bar = times.hour * 60 + times.minute
    for h in HORIZONS:
        fwd = forward_returns(opens, bars, h)
        if fwd.empty:
            print(f"  h={h:>3} min: no data")
            continue
        entries = fwd["net"].to_numpy()
        se = float(entries.std(ddof=1) / np.sqrt(len(entries)))
        # Control: same minute-of-session, random bar, same trade direction.
        # Bars are keyed by minute only. Keying on the bar's own return sign
        # would sample only bars that already moved our way, biasing the
        # control upward and flattering the entries.
        idx_by_min: dict[int, list[int]] = {}
        for i in range(1, len(close) - h):
            idx_by_min.setdefault(int(minute_of_bar[i]), []).append(i)
        ctrl: list[float] = []
        for m, d in zip(
            fwd["minute_of_session"].to_numpy(dtype=int),
            fwd["direction"].to_numpy(dtype=int),
            strict=True,
        ):
            pool = idx_by_min.get(m)
            if not pool:
                continue
            j = int(rng.choice(pool))
            ctrl.append(float(close[j + h] - close[j]) * d - 2.0 * float(spread_pts[j]))
        tstat = float(entries.mean() / se) if se > 0 else float("nan")
        line = (
            f"  h={h:>3} min | entries {_stats(entries)} | t={tstat:>+6.2f} | "
            f"matched ctrl {_stats(np.asarray(ctrl))}"
        )
        if ctrl:
            diff = entries.mean() - float(np.mean(ctrl))
            dse = float(np.sqrt(se**2 + np.std(ctrl, ddof=1) / np.sqrt(len(ctrl)) ** 2))
            line += f" | diff {diff:>+7.2f} +/- {dse:4.2f}"
        print(line)
    print()


def main() -> int:
    bars = pd.read_parquet(BARS)
    for run in RUNS:
        for split in ("training", "testing"):
            opens = load_entries(run, split)
            if opens.empty:
                continue
            report(opens, bars, f"{run.name} / {split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
