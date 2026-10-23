"""End-to-end data pipeline: raw CSVs → cleaned parquet cache.

Usage
-----
    from market_data.loading.pipeline import run_pipeline, build_tick_books
    dfs   = run_pipeline(cfg)
    ticks = build_tick_books(cfg)   # {symbol: TickBook | None}

Bar caches are versioned (``BAR_CACHE_VERSION``) so a full-day rebuild is not
silently skipped in favour of a stale NY-only parquet. Session filtering is
not applied to the feature dataset; ``session`` is an eligibility label.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from market_data.loading.bars import load_bars
from market_data.loading.ticks import TICK_CACHE_VERSION, TickBook, build_tick_book
from market_data.transforms.cleaning import clean
from market_data.transforms.price_grid import (
    configured_price_grid_increment,
    price_grid_identity,
    quantize_bar_prices,
    tick_cache_filename,
)
from market_data.transforms.resample import resample
from market_data.transforms.sessions import add_session_id, add_session_labels

log = logging.getLogger(__name__)

# Native source-timeframe caches are incompatible with M1-resampled v2 caches.
BAR_CACHE_VERSION = "v3-native-timeframes"


def _cache_path(cache_dir: Path, symbol: str, tf: str) -> Path:
    return cache_dir / f"{symbol}_{tf}_{BAR_CACHE_VERSION}.parquet"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(
    cfg: DictConfig,
    force: bool = False,
    datasets: Sequence[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load, resample, clean, label and cache all symbols / timeframes.

    Full-day M1 is kept for every timeframe. ``filter_session`` is not applied.

    Parameters
    ----------
    cfg:
        OmegaConf config (config/base/default.yaml).
    force:
        If True, ignore existing parquet cache and reprocess from CSV.
    """
    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tz = str(cfg.data.tz)
    ny_end = str(cfg.session.end)
    require_native_timeframes = bool(cfg.data.get("require_native_timeframes", False))
    derived_timeframes = cfg.data.get("derived_timeframes", {})
    timestamp_convention = str(cfg.data.get("bar_timestamp_convention", "open")).lower()
    grid_increment = configured_price_grid_increment(cfg)
    grid_identity = f":{price_grid_identity(grid_increment)}" if grid_increment is not None else ""
    if timestamp_convention != "open":
        raise ValueError(
            "timeframe close alignment currently requires open-stamped bars; "
            f"configured convention is {timestamp_convention!r}"
        )
    configured_sets = cfg.data.get("datasets")
    if configured_sets:
        dataset_specs = [
            (
                str(name),
                Path(str(spec.raw_dir)),
                spec.m1_files,
                spec.get("timeframe_files", {}),
            )
            for name, spec in configured_sets.items()
            if datasets is None or str(name) in datasets
        ]
        if datasets is not None:
            missing = set(datasets).difference(str(name) for name in configured_sets)
            if missing:
                raise ValueError(f"requested datasets are not configured: {sorted(missing)}")
    else:
        if datasets is not None and set(datasets) != {"all"}:
            raise ValueError("only the 'all' dataset is configured")
        dataset_specs = [("all", Path(str(cfg.data.raw_dir)), cfg.data.m1_files, {})]

    parts: dict[str, dict[str, list[pd.DataFrame]]] = {
        str(symbol): {str(tf): [] for tf in cfg.data.timeframes} for symbol in cfg.data.symbols
    }
    for dataset_name, root, m1_files, timeframe_files in dataset_specs:
        part_cache = cache_dir / dataset_name
        part_cache.mkdir(parents=True, exist_ok=True)
        for symbol in cfg.data.symbols:
            configured_m1 = timeframe_files.get("M1", {}).get(symbol)
            m1_path = root / str(configured_m1 or m1_files[symbol])
            m1_raw = quantize_bar_prices(load_bars(m1_path), grid_increment)
            source_hashes = {m1_path: _source_sha256(m1_path)}

            def input_hash(path: Path) -> str:
                if path not in source_hashes:
                    source_hashes[path] = _source_sha256(path)
                return source_hashes[path]

            m1_clean = clean(m1_raw, tz=tz)
            m1_full = add_session_labels(m1_clean, tz=tz, ny_end=ny_end)
            m1_full = add_session_id(m1_full)
            for tf in cfg.data.timeframes:
                tf = str(tf)
                cp = _cache_path(part_cache, str(symbol), tf)
                native_file = timeframe_files.get(tf, {}).get(symbol)
                if tf == "M1":
                    source_path = m1_path
                    source_identity = f"native:{input_hash(source_path)}{grid_identity}"
                elif native_file:
                    source_path = root / str(native_file)
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"configured native {tf} source for {symbol} is missing: {source_path}"
                        )
                    source_identity = f"native:{input_hash(source_path)}{grid_identity}"
                else:
                    allowed_derived = [
                        str(value).upper() for value in derived_timeframes.get(symbol, [])
                    ]
                    if require_native_timeframes and tf not in allowed_derived:
                        raise FileNotFoundError(
                            f"native {tf} source for {symbol} is required but not configured; "
                            "declare an intentional M1-derived timeframe explicitly to proceed"
                        )
                    source_path = m1_path
                    source_identity = (
                        f"derived-from-M1:{input_hash(source_path)}:{tf}{grid_identity}"
                    )
                identity_path = Path(f"{cp}.source.sha256")
                cached_identity = (
                    identity_path.read_text(encoding="utf-8").strip()
                    if identity_path.is_file()
                    else None
                )
                if cp.exists() and not force and cached_identity == source_identity:
                    df = pd.read_parquet(cp)
                else:
                    if tf == "M1":
                        df = m1_full.copy()
                    elif native_file:
                        tf_raw = quantize_bar_prices(load_bars(source_path), grid_increment)
                        tf_clean = clean(tf_raw, tz=tz)
                        df = add_session_labels(tf_clean, tz=tz, ny_end=ny_end)
                        df = add_session_id(df)
                    else:
                        # Explicitly absent timeframe sources may be represented
                        # by a derived M1 aggregation. A configured source is
                        # never silently replaced when missing.
                        tf_bars = resample(m1_full, tf)  # type: ignore[arg-type]
                        tf_clean = clean(tf_bars, tz=tz)
                        df = add_session_labels(tf_clean, tz=tz, ny_end=ny_end)
                        df = add_session_id(df)
                    df.to_parquet(cp)
                    identity_path.write_text(source_identity + "\n", encoding="utf-8")
                parts[str(symbol)][str(tf)].append(df)

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol, timeframes in parts.items():
        result[symbol] = {}
        for tf, frames in timeframes.items():
            combined = pd.concat(frames).sort_index()
            if combined.index.has_duplicates:
                raise ValueError(f"overlapping {tf} rows in configured data splits for {symbol}")
            result[symbol][tf] = combined
    return result


def build_tick_books(
    cfg: DictConfig,
    force: bool = False,
    symbols: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> dict[str, TickBook | None]:
    """Build (or load from cache) a ``TickBook`` for each selected symbol.

    By default, include every configured symbol. ``symbols`` narrows the load
    when a caller only consumes a subset, avoiding unnecessary large tick
    arrays. Returns ``{symbol: TickBook | None}``, where ``None`` means the
    tick file was not configured or does not exist. Ticks are stored for the
    full day so overnight replay can quote skipped bars.
    """
    configured_symbols = [str(sym) for sym in cfg.data.symbols]
    selected_symbols = (
        configured_symbols if symbols is None else list(dict.fromkeys(str(sym) for sym in symbols))
    )
    unknown_symbols = set(selected_symbols).difference(configured_symbols)
    if unknown_symbols:
        raise ValueError(f"requested tick symbols are not configured: {sorted(unknown_symbols)}")

    try:
        use_ticks = bool(cfg.costs.use_tick_execution)
    except Exception:
        use_ticks = True

    result: dict[str, TickBook | None] = {}

    if not use_ticks:
        log.info("Tick execution disabled (costs.use_tick_execution=false)")
        for sym in selected_symbols:
            result[sym] = None
        return result

    cache_dir = Path(cfg.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    configured_sets = cfg.data.get("datasets")
    if configured_sets:
        tick_sources = [
            (
                str(name),
                Path(str(spec.raw_dir)),
                dict(spec.get("tick_files", {})),
            )
            for name, spec in configured_sets.items()
            if datasets is None or str(name) in datasets
        ]
        if datasets is not None:
            missing = set(datasets).difference(str(name) for name in configured_sets)
            if missing:
                raise ValueError(f"requested datasets are not configured: {sorted(missing)}")
    else:
        if datasets is not None and set(datasets) != {"all"}:
            raise ValueError("only the 'all' dataset is configured")
        tick_sources = [
            (
                "all",
                Path(str(cfg.data.raw_dir)),
                dict(cfg.data.get("tick_files", {})),
            )
        ]

    grid_increment = configured_price_grid_increment(cfg)
    for sym in selected_symbols:
        books: list[TickBook] = []
        for dataset_name, raw_dir, tick_files in tick_sources:
            tick_file = tick_files.get(sym)
            if not tick_file:
                log.warning("No %s tick file configured for %s", dataset_name, sym)
                continue
            tick_path = raw_dir / str(tick_file)
            if not tick_path.exists():
                log.warning("Tick file not found: %s", tick_path)
                continue
            cache_scope = str(dataset_name) if configured_sets else ""
            cache_p = (
                cache_dir
                / cache_scope
                / tick_cache_filename(sym, grid_increment, TICK_CACHE_VERSION)
            )
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            try:
                book = build_tick_book(
                    path=tick_path,
                    tz=cfg.data.tz,
                    session_start=None,
                    session_end=None,
                    cache_path=cache_p,
                    force=force,
                    price_grid_increment=grid_increment,
                )
                if book is not None and len(book) > 0:
                    books.append(book)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to build TickBook for %s: %s", tick_path, exc)
                result[sym] = None
                books = []
                break
        if books:
            if any(books[i]._ts[-1] >= books[i + 1]._ts[0] for i in range(len(books) - 1)):
                raise ValueError(
                    f"configured tick partitions overlap or are out of order for {sym}"
                )
            import numpy as np

            combined_book = TickBook(
                np.concatenate([book._ts for book in books]),
                np.concatenate([book._bid for book in books]),
                np.concatenate([book._ask for book in books]),
            )
            result[sym] = combined_book
            log.info(
                "TickBook ready: %s (%d ticks, %d partitions)", sym, len(combined_book), len(books)
            )
        elif sym not in result:
            log.warning("No usable tick partitions for %s — using bar-spread fallback", sym)
            result[sym] = None

    return result
