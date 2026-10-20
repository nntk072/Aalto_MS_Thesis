"""Training data materialization and tick-execution preflight helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_data.loading.ticks import TickBook, tick_execution_coverage
from market_data.transforms.sessions import ny_session_mask
from training.setup.environments import strategy_from_cfg


def _publish_obs_memmap(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: Any,
    path: Path,
) -> str | None:
    """Write the model observation matrix once so workers mmap it read-only.

    ``n_envs <= 1`` keeps the private ``to_numpy`` copy. The pandas frame is
    still passed in; this only removes the extra float32 matrix per worker.
    """
    if int(cfg.env.get("n_envs", 1)) <= 1:
        return None
    strategy, _, _ = strategy_from_cfg(cfg)
    from simulation.environment.bundles import observation_features

    array, _ = observation_features(bars, features, strategy.raw_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path)


def _tick_execution_split_schedule(
    index: pd.DatetimeIndex,
    train_mask: pd.Series,
    test_start: str,
    *,
    session_labels: pd.Series | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the observed NY decisions and next-bar quote anchors by split.

    ``TradingEnv`` advances through NY-eligible bars and resolves each step's
    execution/valuation quote at the following market bar. Requiring coverage
    for every overnight M1 row is unrelated to this environment's execution
    schedule and can reject otherwise complete NY-session tick data. The
    returned timestamps describe the observed-bar schedule only; they do not
    define which intervals the CFD provider was expected to publish.
    """
    if session_labels is None:
        ny_mask = ny_session_mask(index).to_numpy(dtype=bool)
    else:
        if not session_labels.index.equals(index):
            raise ValueError("session_labels index must match tick-preflight index")
        ny_mask = session_labels.astype(str).eq("ny").to_numpy(dtype=bool)

    train_mask_values = np.asarray(train_mask.to_numpy(), dtype=bool)
    if len(train_mask_values) != len(index) or not train_mask.index.equals(index):
        raise ValueError("train_mask index must match tick-preflight index")
    test_start_ts = pd.Timestamp(test_start)
    if index.tz is not None:
        if test_start_ts.tzinfo is None:
            test_start_ts = test_start_ts.tz_localize(index.tz)
        else:
            test_start_ts = test_start_ts.tz_convert(index.tz)

    # Match TradingEnv._step_untraced: every eligible NY step consumes the
    # following M1 market quote (clamped to the final available row).
    next_market = np.minimum(np.arange(len(index), dtype=np.int64) + 1, len(index) - 1)
    train_positions = np.flatnonzero(train_mask_values & ny_mask)
    test_positions = np.flatnonzero((index >= test_start_ts) & ny_mask)
    return {
        "train": {
            "decision_positions": train_positions,
            "decision_times": index[train_positions],
            "quote_anchor_positions": next_market[train_positions],
            "quote_anchor_times": index[next_market[train_positions]],
        },
        "test": {
            "decision_positions": test_positions,
            "decision_times": index[test_positions],
            "quote_anchor_positions": next_market[test_positions],
            "quote_anchor_times": index[next_market[test_positions]],
        },
    }


def _tick_execution_split_coverage(
    tickbook: TickBook | None,
    index: pd.DatetimeIndex,
    train_mask: pd.Series,
    test_start: str,
    *,
    session_labels: pd.Series | None = None,
    max_latency_ns_exclusive: int,
    max_quote_wait_ms: int,
) -> dict[str, dict[str, Any]]:
    """Preflight tick quotes at the timestamps actually used by NY env steps."""
    schedule = _tick_execution_split_schedule(
        index, train_mask, test_start, session_labels=session_labels
    )
    options = {
        "max_latency_ns_exclusive": max_latency_ns_exclusive,
        "max_quote_wait_ms": max_quote_wait_ms,
    }
    return {
        split: tick_execution_coverage(tickbook, rows["quote_anchor_times"], **options)
        for split, rows in schedule.items()
    }
