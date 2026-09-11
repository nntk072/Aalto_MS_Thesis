"""End-to-end data pipeline: raw CSVs → cleaned parquet cache.

Usage
-----
    from quant_rl.data.pipeline import run_pipeline, build_tick_books
    dfs   = run_pipeline(cfg)
    ticks = build_tick_books(cfg)   # {symbol: TickBook | None}

Bar caches are versioned (``BAR_CACHE_VERSION``) so a full-day rebuild is not
silently skipped in favour of a stale NY-only parquet. Session filtering is
not applied to the feature dataset; ``session`` is an eligibility label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from .clean import clean
from .loader import load_bars
from .resample import resample
from .session import add_session_id, add_session_labels
from .ticks import TickBook, build_tick_book

log = logging.getLogger(__name__)

# Busts unversioned ``{symbol}_{tf}.parquet`` NY-only caches.
BAR_CACHE_VERSION = "v2-full-day"


def _cache_path(cache_dir: Path, symbol: str, tf: str) -> Path:
    return cache_dir / f"{symbol}_{tf}_{BAR_CACHE_VERSION}.parquet"


def run_pipeline(cfg: DictConfig, force: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    """Load, resample, clean, label and cache all symbols / timeframes.

    Full-day M1 is kept for every timeframe. ``filter_session`` is not applied.

    Parameters
    ----------
    cfg:
        OmegaConf config (quant_rl/config/default.yaml).
    force:
        If True, ignore existing parquet cache and reprocess from CSV.
    """
    root = Path(cfg.data.raw_dir)
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tz = str(cfg.data.tz)
    ny_end = str(cfg.session.end)

    result: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in cfg.data.symbols:
        result[symbol] = {}

        m1_file = cfg.data.m1_files[symbol]
        m1_path = root / m1_file
        m1_raw = load_bars(m1_path)
        m1_clean = clean(m1_raw, tz=tz)
        m1_full = add_session_labels(m1_clean, tz=tz, ny_end=ny_end)
        m1_full = add_session_id(m1_full)

        for tf in cfg.data.timeframes:
            cp = _cache_path(cache_dir, symbol, tf)
            if cp.exists() and not force:
                df = pd.read_parquet(cp)
                result[symbol][tf] = df
                continue

            if tf == "M1":
                df = m1_full.copy()
            else:
                tf_bars = resample(m1_full, tf)
                tf_clean = clean(tf_bars, tz=tz)
                df = add_session_labels(tf_clean, tz=tz, ny_end=ny_end)
                df = add_session_id(df)

            df.to_parquet(cp)
            result[symbol][tf] = df

    return result


def build_tick_books(
    cfg: DictConfig,
    force: bool = False,
) -> dict[str, TickBook | None]:
    """Build (or load from cache) a ``TickBook`` for each symbol.

    Returns a dict ``{symbol: TickBook | None}`` where ``None`` means the
    tick file was not configured or does not exist. Ticks are stored for the
    full day so overnight replay can quote skipped bars.
    """
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

    raw_dir = Path(cfg.data.raw_dir)
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tick_files: dict[str, str] = {}
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

        cache_p = cache_dir / f"{sym}_ticks_{BAR_CACHE_VERSION}.parquet"
        try:
            result[sym] = build_tick_book(
                path=tick_path,
                tz=cfg.data.tz,
                session_start=None,
                session_end=None,
                cache_path=cache_p,
                force=force,
            )
            tick_book = result[sym]
            if tick_book is not None:
                log.info("TickBook ready: %s  (%d ticks)", sym, len(tick_book))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to build TickBook for %s: %s — fallback", sym, exc)
            result[sym] = None

    return result
