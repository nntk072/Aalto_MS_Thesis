"""Load raw MT5 bar CSV files into DataFrames.

File format (tab-separated):
  <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>

Date format: YYYY.MM.DD, time: HH:MM:SS
SPREAD is in price points.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_BAR_COLS = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]
_TICK_COLS = ["date", "time", "bid", "ask", "last", "volume", "flags"]


def load_bars(path: str | Path) -> pd.DataFrame:
    """Parse a raw MT5 bar CSV into a DataFrame with a UTC-aware DatetimeIndex.

    The returned index name is ``datetime`` and the timezone is left as read
    (naive) because the caller applies the broker timezone via the config.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=_BAR_COLS,
        dtype={
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "tickvol": "int64",
            "vol": "int64",
            "spread": "float64",
        },
        na_values=[""],
    )
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S"
    )
    df = df.drop(columns=["date", "time"])
    df = df.set_index("datetime")
    df = df.sort_index()
    return df


def load_ticks(path: str | Path) -> pd.DataFrame:
    """Parse a raw MT5 tick CSV (BID/ASK) into a DataFrame.

    Tick time includes milliseconds (``HH:MM:SS.fff``), unlike bar files.

    Warning
    -------
    Loads the **entire file into memory at once**.  MT5 tick exports can be
    multiple GB — for those, use :func:`iter_ticks_chunks` instead (streams
    bounded chunks, used by ``quant_rl.data.ticks.build_tick_book``).
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=_TICK_COLS,
        dtype={"bid": "float64", "ask": "float64", "last": "float64", "volume": "float64"},
        na_values=[""],
    )
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S.%f"
    )
    df = df.drop(columns=["date", "time"])
    df = df.set_index("datetime")
    df = df.sort_index()
    return df


def iter_ticks_chunks(path: str | Path, chunksize: int = 2_000_000):
    """Stream a raw MT5 tick CSV in bounded-size chunks.

    Only the ``date``, ``time``, ``bid``, ``ask`` columns are parsed —
    ``last``/``volume``/``flags`` are skipped entirely via ``usecols`` — and
    ``bid``/``ask`` are read directly as ``float32``.  This keeps peak memory
    to roughly one chunk's worth of data instead of the whole file, which
    matters for multi-GB MT5 tick exports (a naive full-file load can OOM a
    memory-constrained VM such as WSL2).

    Yields
    ------
    Raw ``DataFrame`` chunks with columns ``["date", "time", "bid", "ask"]``
    (still as string ``date``/``time`` — the caller parses datetimes itself
    so it can carry forward-fill state across chunk boundaries).
    """
    path = Path(path)
    reader = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=_TICK_COLS,
        usecols=["date", "time", "bid", "ask"],
        dtype={"bid": "float32", "ask": "float32"},
        na_values=[""],
        chunksize=chunksize,
        engine="c",
    )
    yield from reader
