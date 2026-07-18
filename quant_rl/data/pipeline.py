"""End-to-end data pipeline: raw CSVs → cleaned parquet cache.

Usage
-----
    from quant_rl.data.pipeline import run_pipeline, build_tick_books
    dfs   = run_pipeline(cfg)
    ticks = build_tick_books(cfg)   # {symbol: TickBook | None}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from .loader import load_bars
from .resample import resample, build_all_timeframes
from .clean import clean
from .session import filter_session, add_session_id
from .ticks import TickBook, build_tick_book

log = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, symbol: str, tf: str) -> Path:
    return cache_dir / f"{symbol}_{tf}.parquet"


def run_pipeline(cfg: DictConfig, force: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    """Load, resample, clean, filter and cache all symbols / timeframes.

    Parameters
    ----------
    cfg:
        OmegaConf config (quant_rl/config/default.yaml).
    force:
        If True, ignore existing parquet cache and reprocess from CSV.

    Returns
    -------
    Nested dict: ``result[symbol][tf]`` → cleaned, session-filtered DataFrame.
    """
    root = Path(cfg.data.raw_dir)
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in cfg.data.symbols:
        result[symbol] = {}

        # --- load true M1 source ---
        m1_file = cfg.data.m1_files[symbol]
        m1_path = root / m1_file
        m1_raw = load_bars(m1_path)
        m1_clean = clean(m1_raw, tz=cfg.data.tz)
        m1_session = filter_session(m1_clean, start=cfg.session.start, end=cfg.session.end)
        m1_session = add_session_id(m1_session)

        for tf in cfg.data.timeframes:
            cp = _cache_path(cache_dir, symbol, tf)
            if cp.exists() and not force:
                df = pd.read_parquet(cp)
                result[symbol][tf] = df
                continue

            if tf == "M1":
                df = m1_session.copy()
            else:
                # Resample *before* session filter so we don't lose partial bars
                tf_bars = resample(m1_clean, tf)
                tf_clean = clean(tf_bars, tz=cfg.data.tz)
                df = filter_session(tf_clean, start=cfg.session.start, end=cfg.session.end)
                df = add_session_id(df)

            df.to_parquet(cp)
            result[symbol][tf] = df

    return result


def build_tick_books(
    cfg: DictConfig,
    force: bool = False,
) -> dict[str, "TickBook | None"]:
    """Build (or load from cache) a ``TickBook`` for each symbol.

    Returns a dict ``{symbol: TickBook | None}`` where ``None`` means the
    tick file was not configured or does not exist.

    Parameters
    ----------
    cfg:
        OmegaConf config.  Reads ``data.tick_files``, ``data.raw_dir``,
        ``data.tz``, ``session.*``, ``data.cache_dir``, and
        ``costs.use_tick_execution``.
    force:
        Rebuild the tick parquet caches even if they already exist.
    """
    # Honour kill-switch
    try:
        use_ticks = bool(cfg.costs.use_tick_execution)
    except Exception:
        use_ticks = True

    result: dict[str, TickBook | None] = {}

    if not use_ticks:
        log.info("Tick execution disabled (costs.use_tick_execution=false)")
        for sym in cfg.data.symbols:
            result[sym] = None
        return result

    raw_dir   = Path(cfg.data.raw_dir)
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tick_files: dict = {}
    try:
        tick_files = dict(cfg.data.tick_files)
    except Exception:
        pass

    for sym in cfg.data.symbols:
        tick_file = tick_files.get(sym)
        if not tick_file:
            log.warning("No tick file configured for %s — using bar-spread fallback", sym)
            result[sym] = None
            continue

        tick_path = raw_dir / tick_file
        if not tick_path.exists():
            log.warning("Tick file not found: %s — using bar-spread fallback", tick_path)
            result[sym] = None
            continue

        cache_p = cache_dir / f"{sym}_ticks.parquet"
        try:
            result[sym] = build_tick_book(
                path=tick_path,
                tz=cfg.data.tz,
                session_start=cfg.session.start,
                session_end=cfg.session.end,
                cache_path=cache_p,
                force=force,
            )
            log.info("TickBook ready: %s  (%d ticks)", sym, len(result[sym]))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to build TickBook for %s: %s — fallback", sym, exc)
            result[sym] = None

    return result
