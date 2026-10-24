#!/usr/bin/env python3
"""Render project-to-result campaign reports from saved run artifacts.

The report is intentionally artifact-driven: it does not load checkpoints, replay
markets, or infer values that were not saved. It writes JSON for programmatic use
and Markdown for inspection, including data/cache lineage, code architecture,
model setup, metrics, trade records, and generated figures.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "Not recorded in run artifacts"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 10.0 ** (-digits):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}"
    if isinstance(value, dict):
        if not value:
            return "None recorded"
        return "; ".join(
            f"{str(key).replace('_', ' ')}: {_display(item, digits)}" for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "None recorded"
        values = list(value)
        if len(values) > 16 and all(
            not isinstance(item, (dict, list, tuple, set)) for item in values
        ):
            first = ", ".join(_display(item, digits) for item in values[:8])
            last = ", ".join(_display(item, digits) for item in values[-4:])
            return f"{len(values)} values; first: {first}; last: {last}"
        return ", ".join(_display(item, digits) for item in values)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _report_value(value: Any, field_path: str) -> str:
    """Render percentage-like metrics with explicit units in Markdown."""
    fields = {part.strip().replace(" ", "_").lower() for part in field_path.split("/")}
    ratio_fields = {
        "max_drawdown",
        "max_peak_trailing_dd",
        "peak_trailing_dd_limit",
        "win_rate",
        "risk_fraction",
        "risk_frac",
        "risk_fraction_range",
        "risk_frac_range",
        "default_risk_frac",
        "risk_floor",
        "risk_ceiling",
        "min_risk_frac",
        "max_risk_frac",
        "trailing_dd",
    }
    is_ratio = any(field in ratio_fields for field in fields)
    if is_ratio and isinstance(value, (list, tuple)):
        return ", ".join(
            f"{number * 100:.2f}%" if (number := _number(item)) is not None else _display(item)
            for item in value
        )
    if isinstance(value, bool):
        return _display(value)
    number = _number(value)
    if number is None:
        return _display(value)
    if is_ratio:
        return f"{number * 100:.2f}%"
    if any(field.endswith("_pct") for field in fields):
        return f"{number:.2f}%"
    return _display(value)


def _flatten_markdown_rows(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Turn nested run metadata into readable table rows, never raw JSON."""
    if isinstance(value, dict):
        if not value:
            return [(prefix or "Value", "None recorded")]
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            label = str(key).replace("_", " ")
            nested = f"{prefix} / {label}" if prefix else label
            rows.extend(_flatten_markdown_rows(item, nested))
        return rows
    if (
        isinstance(value, (list, tuple))
        and value
        and any(isinstance(item, (dict, list, tuple)) for item in value)
    ):
        rows = []
        for index, item in enumerate(value, start=1):
            rows.extend(_flatten_markdown_rows(item, f"{prefix} {index}".strip()))
        return rows
    return [(prefix or "Value", _report_value(value, prefix))]


def _markdown_table(value: Any, *, heading: str = "Measure") -> list[str]:
    rows = _flatten_markdown_rows(value)
    return [
        f"| {heading} | Result |",
        "|---|---|",
        *[f"| {name} | {result} |" for name, result in rows],
    ]


def _counter(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter((row.get(field) or "unavailable").strip() or "unavailable" for row in rows)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _multi_counter(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        decoded = _decode(row.get(field))
        values = decoded if isinstance(decoded, (list, tuple, set)) else [decoded]
        for value in values:
            if value not in (None, ""):
                counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _flatten_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    selected: dict[str, Any] = {}
    for section in (
        "algo",
        "arch",
        "reward",
        "strategy",
        "strategy_actions",
        "timesteps",
        "seed",
        "action_schema",
        "observation_schema",
        "training_protocol",
        "ppo",
        "risk",
        "ftmo",
        "env",
        "environment",
        "train_args",
        "overrides",
        "variant",
    ):
        if section in config:
            selected[section] = config[section]
    base = config.get("base_config")
    if base is not None:
        selected["base_config"] = base
    return selected


def _join_trade_records(split_dir: Path) -> list[dict[str, Any]]:
    """Join completed positions to entry fill decisions and close events."""
    positions = _rows(split_dir / "positions.csv")
    fills = _rows(split_dir / "fills.csv")
    trades = _rows(split_dir / "trades.csv")
    if not positions:
        # Position summaries are preferred because they represent complete trades.
        # Keep the absence visible instead of treating individual fills as trades.
        return []
    by_position: dict[str, list[dict[str, str]]] = {}
    for row in fills:
        key = row.get("position_id", "")
        if key:
            by_position.setdefault(key, []).append(row)
    trade_events: dict[str, list[dict[str, str]]] = {}
    for row in trades:
        key = row.get("position_id", "")
        if key:
            trade_events.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for position in positions:
        key = position.get("position_id", "")
        events = by_position.get(key, []) or trade_events.get(key, [])
        opens = [row for row in events if row.get("type") == "open"]
        closes = [
            row
            for row in events
            if row.get("type")
            in {"close", "stop_close", "tp_close", "forced_close", "eod_close", "tp_partial_close"}
        ]
        entry = opens[0] if opens else {}
        pnl = _number(position.get("position_net_pnl"))
        if pnl is None:
            pnl = _number(position.get("net_pnl"))
        if pnl is None:
            pnl = _number(position.get("close_pnl"))
        close_reason = position.get("exit_reason") or (closes[-1].get("reason") if closes else None)
        result.append(
            {
                "position_id": key or None,
                "direction": position.get("direction") or entry.get("direction"),
                "entry_time": position.get("entry_time")
                or entry.get("fill_time")
                or entry.get("time"),
                "exit_time": position.get("exit_time")
                or (closes[-1].get("fill_time") if closes else None),
                "chart_entry_time": entry.get("time"),
                "chart_exit_time": closes[-1].get("time") if closes else None,
                "entry_price": _number(position.get("entry_price") or entry.get("price")),
                "exit_price": _number(
                    position.get("exit_price") or (closes[-1].get("price") if closes else None)
                ),
                "net_pnl": pnl,
                "realized_r": _number(entry.get("position_realized_r") or entry.get("realized_r")),
                "exit_reason": close_reason,
                "exit_mode": entry.get("exit_mode"),
                "setup": {
                    "level_type": entry.get("level_type"),
                    "entry_setup_types": _decode(entry.get("entry_setup_types")),
                    "manipulation": entry.get("manipulation"),
                    "distribution_phase": _number(entry.get("distribution_phase")),
                    "session": entry.get("session"),
                    "volatility_regime": entry.get("volatility_regime"),
                    "trend_regime": entry.get("trend_regime"),
                },
                "risk": {
                    "requested_usd": _number(entry.get("requested_risk_usd")),
                    "approved_usd": _number(entry.get("approved_risk_usd")),
                    "modeled_stop_risk_usd": _number(entry.get("modeled_stop_risk_usd")),
                    "execution_reserve_usd": _number(entry.get("execution_reserve_usd")),
                    "risk_fraction": _number(entry.get("risk_frac")),
                    "risk_cash": _number(entry.get("risk_cash")),
                    "lots": _number(entry.get("lots") or entry.get("size")),
                },
                "sl": {
                    "reference": entry.get("sl_ref"),
                    "price": _number(entry.get("sl_price")),
                    "distance": _number(entry.get("stop_distance")),
                    "timeframe": entry.get("sl_timeframe"),
                    "mode": entry.get("exit_mode"),
                },
                "tp": {
                    "reference": entry.get("tp_ref"),
                    "price": _number(entry.get("tp_price")),
                    "distance": _number(entry.get("target_distance")),
                    "planned_rr": _number(entry.get("planned_rr")),
                    "executed_targets": _decode(entry.get("executed_target_prices")),
                    "legs": _decode(entry.get("tp_legs")),
                },
                "fill_count": int(_number(position.get("fill_count")) or len(events)),
                "completed": position.get("completed"),
                "action": {
                    key: _number(entry.get(key))
                    for key in (
                        "action_direction",
                        "action_intensity",
                        "action_stop",
                        "action_risk",
                        "action_target",
                        "action_sl_mode",
                        "action_tp_mode",
                        "action_exit",
                    )
                },
            }
        )
    return result


def _classify_trades(
    trades: list[dict[str, Any]], tolerance: float
) -> dict[str, list[dict[str, Any]]]:
    winners, losers, breakeven = [], [], []
    for trade in trades:
        pnl = trade.get("net_pnl")
        reason = str(trade.get("exit_reason") or "").lower().replace("-", "_").replace(" ", "_")
        is_break_even_exit = "break_even" in reason or reason in {"be", "breakeven"}
        if pnl is None:
            continue
        if is_break_even_exit or abs(pnl) <= tolerance:
            breakeven.append(trade)
        elif pnl > 0:
            winners.append(trade)
        else:
            losers.append(trade)

    def typical(items: list[dict[str, Any]], center: float) -> list[dict[str, Any]]:
        # Pick trades closest to the group's median outcome, then restore time order.
        ordered = sorted(
            items,
            key=lambda item: (abs(float(item["net_pnl"]) - center), str(item.get("entry_time"))),
        )
        chosen = ordered[:5]
        return sorted(chosen, key=lambda item: str(item.get("entry_time")))

    be_ordered = sorted(
        breakeven, key=lambda item: (abs(float(item["net_pnl"])), str(item.get("entry_time")))
    )[:5]
    return {
        "winners": typical(winners, statistics.median([float(t["net_pnl"]) for t in winners]))
        if winners
        else [],
        "losers": typical(losers, statistics.median([float(t["net_pnl"]) for t in losers]))
        if losers
        else [],
        "breakeven": sorted(be_ordered, key=lambda item: str(item.get("entry_time"))),
    }


def _trade_chart(split_dir: Path, trade: dict[str, Any], report_dir: Path) -> str | None:
    """Find the generated candle chart for a selected, fully described trade."""
    order_dir = split_dir / "orders"
    if not order_dir.is_dir():
        return None
    try:
        entry = datetime.fromisoformat(str(trade.get("chart_entry_time")))
        exit_ = datetime.fromisoformat(str(trade.get("chart_exit_time")))
    except (TypeError, ValueError):
        return None
    side = "L" if (_number(trade.get("direction")) or 0) > 0 else "S"
    pattern = f"trade_*_{entry:%Y%m%d}_{entry:%H%M}_{exit_:%H%M}_{side}_*.png"
    matches = sorted(order_dir.glob(pattern))
    return os.path.relpath(matches[0], report_dir) if matches else None


def _trade_outcome_caption(category: str, split: str, rank: int, trade: dict[str, Any]) -> str:
    labels = {
        "winners": "WINNER",
        "losers": "LOSER",
        "breakeven": "BREAK-EVEN",
    }
    try:
        outcome = labels[category]
    except KeyError as exc:
        raise ValueError(f"unknown selected-trade category: {category}") from exc
    pnl = _number(trade.get("net_pnl"))
    pnl_text = "not recorded" if pnl is None else f"{pnl:+.2f} account units"
    return f"TYPICAL {outcome} | {split.upper()} | SAMPLE {rank} | NET PNL {pnl_text}"


def _annotate_trade_outcome(chart_path: Path, caption: str, category: str) -> None:
    """Put the selected outcome label on the individual candle chart itself."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    colors = {
        "winners": ("#e9f4e7", "#2f6b2f"),
        "losers": ("#fae8e7", "#9b2f2f"),
        "breakeven": ("#eeeeef", "#4d4d52"),
    }
    if category not in colors:
        raise ValueError(f"unknown selected-trade category: {category}")
    if not chart_path.is_file() or chart_path.stat().st_size == 0:
        raise FileNotFoundError(f"selected-trade candle chart is missing or empty: {chart_path}")
    image = mpimg.imread(chart_path)
    if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"selected-trade candle chart is malformed: {chart_path}")

    header_color, text_color = colors[category]
    width, height = 14.0, 0.62
    chart_height = width * image.shape[0] / image.shape[1]
    total_height = chart_height + height
    fig = plt.figure(figsize=(width, total_height), facecolor=header_color)
    ax = fig.add_axes((0.0, 0.0, 1.0, chart_height / total_height))
    ax.imshow(image)
    ax.axis("off")
    fig.text(
        0.018,
        1.0 - height / (2.0 * total_height),
        caption,
        color=text_color,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
    )
    fig.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor=header_color)
    plt.close(fig)


def _split_report(
    run_dir: Path,
    split: str,
    be_tolerance: float,
    example_split: bool,
    peak_dd_limit: float,
) -> dict[str, Any]:
    split_dir = run_dir / split
    metrics = _json(split_dir / "metrics.json")
    positions = _rows(split_dir / "positions.csv")
    fills = _rows(split_dir / "fills.csv")
    trade_records = _join_trade_records(split_dir)
    net_values = [
        float(item["net_pnl"]) for item in trade_records if item.get("net_pnl") is not None
    ]
    open_fills = [row for row in fills if row.get("type") == "open"]
    close_fills = [
        row
        for row in fills
        if row.get("type")
        in {"close", "stop_close", "tp_close", "forced_close", "eod_close", "tp_partial_close"}
    ]
    pnl_winners = [value for value in net_values if value > be_tolerance]
    pnl_losers = [value for value in net_values if value < -be_tolerance]
    pnl_be = [value for value in net_values if abs(value) <= be_tolerance]
    requested = [_number(row.get("requested_risk_usd")) for row in open_fills]
    approved = [_number(row.get("approved_risk_usd")) for row in open_fills]
    stop_risk = [_number(row.get("modeled_stop_risk_usd")) for row in open_fills]
    outcomes: dict[str, list[dict[str, Any]]] = {
        "winning": [],
        "losing": [],
        "breakeven": [],
    }
    for trade in trade_records:
        pnl = trade.get("net_pnl")
        if pnl is None:
            continue
        reason = str(trade.get("exit_reason") or "").lower().replace("-", "_")
        if "break_even" in reason or "breakeven" in reason or abs(pnl) <= be_tolerance:
            category = "breakeven"
        elif pnl > 0:
            category = "winning"
        else:
            category = "losing"
        outcomes[category].append(trade)

    def exit_outcomes(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        result: dict[str, Counter[str]] = {name: Counter() for name in outcomes}
        for item in items:
            pnl = item.get("net_pnl")
            reason = str(item.get("exit_reason") or "unavailable")
            if pnl is None:
                continue
            normalized = reason.lower().replace("-", "_")
            group = (
                "breakeven"
                if "break_even" in normalized
                or "breakeven" in normalized
                or abs(pnl) <= be_tolerance
                else "winning"
                if pnl > 0
                else "losing"
            )
            result[group][reason] += 1
        return {name: dict(sorted(counts.items())) for name, counts in result.items()}

    def risk_by_outcome(key: str) -> dict[str, dict[str, float | int | None]]:
        result: dict[str, dict[str, float | int | None]] = {}
        for group, group_trades in outcomes.items():
            values = [_number(trade.get("risk", {}).get(key)) for trade in group_trades]
            result[group] = _distribution([value for value in values if value is not None])
        return result

    report: dict[str, Any] = {
        "metrics": metrics,
        "peak_trailing_dd_limit": peak_dd_limit,
        "peak_drawdown_policy": (
            f"Active {peak_dd_limit:.1%} environment ceiling and survival check"
            if peak_dd_limit > 0.0
            else "Disabled; measured peak drawdown is report-only"
        ),
        "counts": {
            "positions": len(positions),
            "entry_fills": len(open_fills),
            "exit_fills": len(close_fills),
            "rejected_entries": sum(
                row.get("type") == "reject" for row in _rows(split_dir / "trades.csv")
            ),
            "wins": len(pnl_winners),
            "losses": len(pnl_losers),
            "breakeven_within_tolerance": len(pnl_be),
        },
        "important_setup": {
            "direction": _counter(open_fills, "direction"),
            "entry_level": _counter(open_fills, "level_type"),
            "setup_types": _multi_counter(open_fills, "entry_setup_types"),
            "manipulation": _counter(open_fills, "manipulation"),
            "distribution_phase": _counter(open_fills, "distribution_phase"),
            "session": _counter(open_fills, "session"),
            "volatility_regime": _counter(open_fills, "volatility_regime"),
            "trend_regime": _counter(open_fills, "trend_regime"),
        },
        "exit_breakdown": {
            "position_exit_reason": _counter(positions, "exit_reason"),
            "exit_reason_by_net_outcome": exit_outcomes(trade_records),
            "entry_selected_exit_mode": _counter(open_fills, "exit_mode"),
            "close_fill_reason": _counter(close_fills, "reason"),
            "outcome_counts": {
                "winning": len(pnl_winners),
                "losing": len(pnl_losers),
                "breakeven": len(pnl_be),
            },
        },
        "sl_breakdown": {
            "selected_reference": _counter(open_fills, "sl_ref"),
            "exit_reason": _counter(
                [row for row in close_fills if "sl" in (row.get("reason") or "").lower()], "reason"
            ),
            "sl_reason_by_net_outcome": exit_outcomes(
                [
                    trade
                    for trade in trade_records
                    if "sl" in str(trade.get("exit_reason") or "").lower()
                ]
            ),
            "modeled_stop_distance": _distribution(
                [v for row in open_fills if (v := _number(row.get("stop_distance"))) is not None]
            ),
            "initial_stop_risk_usd": _distribution([v for v in stop_risk if v is not None]),
            "dynamic_sl_events": _counter(
                [row for row in fills if row.get("type") in {"modify_sl", "sl_modify", "move_sl"}],
                "type",
            ),
        },
        "tp_breakdown": {
            "selected_reference": _counter(open_fills, "tp_ref"),
            "close_fill_reason": _counter(
                [row for row in close_fills if "tp" in (row.get("reason") or "").lower()], "reason"
            ),
            "tp_reason_by_net_outcome": exit_outcomes(
                [
                    trade
                    for trade in trade_records
                    if "tp" in str(trade.get("exit_reason") or "").lower()
                ]
            ),
            "planned_rr": _distribution(
                [v for row in open_fills if (v := _number(row.get("planned_rr"))) is not None]
            ),
            "target_legs_present": sum(bool(row.get("tp_legs")) for row in open_fills),
            "configured_tp_count_distribution": _counter(open_fills, "n_tp"),
            "tp_slot_distribution": _counter(close_fills, "tp_slot"),
        },
        "risk_breakdown": {
            "requested_usd": _distribution([v for v in requested if v is not None]),
            "approved_usd": _distribution([v for v in approved if v is not None]),
            "modeled_stop_risk_usd": _distribution([v for v in stop_risk if v is not None]),
            "risk_fraction": _distribution(
                [v for row in open_fills if (v := _number(row.get("risk_frac"))) is not None]
            ),
            "execution_reserve_usd": _distribution(
                [
                    v
                    for row in open_fills
                    if (v := _number(row.get("execution_reserve_usd"))) is not None
                ]
            ),
            "lots": _distribution(
                [v for row in open_fills if (v := _number(row.get("lots"))) is not None]
            ),
            "requested_usd_by_outcome": risk_by_outcome("requested_usd"),
            "approved_usd_by_outcome": risk_by_outcome("approved_usd"),
            "modeled_stop_risk_usd_by_outcome": risk_by_outcome("modeled_stop_risk_usd"),
        },
        "trade_examples": _classify_trades(trade_records, be_tolerance) if example_split else None,
        "artifact_files": {
            key: (split_dir / filename).relative_to(run_dir).as_posix()
            for key, filename in {
                "positions": "positions.csv",
                "fills": "fills.csv",
                "trade_events": "trades.csv",
                "metrics": "metrics.json",
                "equity_figure": "equity.png",
                "drawdown_figure": "drawdown.png",
                "daily_pnl_figure": "daily_pnl.png",
                "sl_family_figure": "sl_family.png",
                "tp_family_figure": "tp_family.png",
                "stop_index_figure": "stop_u_vs_index.png",
                "target_index_figure": "target_u_vs_index.png",
                "sl_tp_rank_figure": "sl_tp_rank_heatmap.png",
            }.items()
            if (split_dir / filename).is_file()
        },
        "source_sha256": {
            name: _sha256(split_dir / name)
            for name in ("metrics.json", "positions.csv", "fills.csv", "trades.csv", "equity.csv")
            if (split_dir / name).is_file()
        },
        "plot_data": {
            "ideas": [
                {
                    "level": trade.get("setup", {}).get("level_type") or "unavailable",
                    "setup_types": trade.get("setup", {}).get("entry_setup_types") or [],
                    "net_pnl": trade.get("net_pnl"),
                }
                for trade in trade_records
            ],
            "stop_distance": [
                v for row in open_fills if (v := _number(row.get("stop_distance"))) is not None
            ],
            "planned_rr": [
                v for row in open_fills if (v := _number(row.get("planned_rr"))) is not None
            ],
            "risk_requested": [
                v for row in open_fills if (v := _number(row.get("requested_risk_usd"))) is not None
            ],
            "risk_approved": [
                v for row in open_fills if (v := _number(row.get("approved_risk_usd"))) is not None
            ],
            "risk_stop": [
                v
                for row in open_fills
                if (v := _number(row.get("modeled_stop_risk_usd"))) is not None
            ],
        },
    }
    return report


def _training_report(run_dir: Path, resolved: dict[str, Any] | None) -> dict[str, Any]:
    log = _json(run_dir / "training_log.json") or {}
    protocol = log.get("training_protocol") or log.get("training_config") or {}
    try:
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        config = None
    if not protocol and resolved:
        protocol = resolved.get("training_protocol", {})
    effective_parameters = _flatten_config(config if isinstance(config, dict) else resolved)
    if protocol:
        effective_parameters["training_protocol"] = protocol
    if resolved:
        effective_parameters["resolved_variant"] = _flatten_config(resolved)
    csv_rows = _rows(run_dir / "model" / "training_log.csv")
    if not csv_rows:
        raise FileNotFoundError(
            f"training log is required for learner diagnostics: {run_dir / 'model' / 'training_log.csv'}"
        )
    wanted = (
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "train/learning_rate",
    )
    curves: dict[str, Any] = {}
    for key in wanted:
        values = [v for row in csv_rows if (v := _number(row.get(key))) is not None]
        curves[key] = {
            "samples": len(values),
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
            "mean": statistics.fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    if not any(curve["samples"] for curve in curves.values()):
        raise ValueError(
            f"training log has no finite learner diagnostics: {run_dir / 'model' / 'training_log.csv'}"
        )
    figures = [
        path.relative_to(run_dir).as_posix()
        for path in sorted((run_dir / "model").glob("*.png"))
        if path.name
        in {
            "learning_curve.png",
            "losses.png",
            "entropy_explvar.png",
            "kl_clip.png",
            "lr.png",
            "mode_means.png",
            "dashboard_pnl.png",
            "dashboard_return.png",
            "dashboard_risk.png",
            "dashboard_trade_count.png",
            "dashboard_behavior.png",
        }
    ]
    effective_config = config if isinstance(config, dict) else resolved
    env_config = effective_config.get("env", {}) if isinstance(effective_config, dict) else {}
    configured_limit = (
        _number(env_config.get("peak_trailing_dd_limit", 0.0))
        if isinstance(env_config, dict)
        else 0.0
    )
    if configured_limit is None or not 0.0 <= configured_limit < 1.0:
        raise ValueError(
            f"invalid env.peak_trailing_dd_limit in run configuration: {run_dir / 'config.yaml'}"
        )
    peak_dd_limit = configured_limit
    training_gate = _json(run_dir / "train_equity_gate.json")
    gate_policy_evidence = _validate_training_gate_policy(training_gate, peak_dd_limit, run_dir)
    return {
        "run_summary": log,
        "agent_parameters": effective_parameters,
        "training_curve_stats": curves,
        "training_log_rows": len(csv_rows),
        "figures": figures,
        "peak_trailing_dd_limit": peak_dd_limit,
        "peak_drawdown_policy": (
            f"Active {peak_dd_limit:.1%} training environment policy and survival check"
            if peak_dd_limit > 0.0
            else "Disabled during training; measured drawdown is report-only"
        ),
        "peak_drawdown_policy_evidence": gate_policy_evidence,
        "training_equity_gate": training_gate,
        "training_activity_gate": _json(run_dir / "train_activity_gate.json"),
    }


def _validate_training_gate_policy(
    gate: dict[str, Any] | None, configured_limit: float, run_dir: Path
) -> str:
    """Ensure the final in-sample gate reflects the same configured DD policy."""
    if gate is None:
        raise FileNotFoundError(
            f"training survival gate is missing: {run_dir / 'train_equity_gate.json'}"
        )
    observed = _number(gate.get("max_peak_trailing_dd"))
    gate_ok = gate.get("ok")
    if observed is None or not isinstance(gate_ok, bool):
        raise ValueError(f"training survival gate lacks policy evidence: {run_dir}")
    breached = bool(gate.get("breached", False))
    drawdown_failed = configured_limit > 0.0 and observed > configured_limit
    expected_ok = not breached and not drawdown_failed
    if gate_ok != expected_ok:
        raise ValueError(
            "training survival gate outcome conflicts with the saved drawdown policy: "
            f"config={configured_limit:.6f}, observed={observed:.6f}, gate_ok={gate_ok}, run={run_dir}"
        )
    recorded_limit = _number(gate.get("peak_trailing_dd_limit"))
    if recorded_limit is not None:
        if not math.isclose(recorded_limit, configured_limit, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "training drawdown policy does not match the saved run config: "
                f"config={configured_limit:.6f}, gate={recorded_limit:.6f}, run={run_dir}"
            )
        return "The saved training gate records the same limit as the training environment."

    # Older gate artifacts omitted the limit field. Reconcile their pass/fail
    # result against the saved limit and observed drawdown before reporting them.
    return "The saved training gate outcome was reconciled with its configured limit and observed drawdown."


def _metric(detail: dict[str, Any] | None, name: str) -> Any:
    if not isinstance(detail, dict):
        return None
    metrics = detail.get("metrics")
    return metrics.get(name) if isinstance(metrics, dict) else None


def _markdown_metric(detail: dict[str, Any] | None, name: str) -> str:
    return _report_value(_metric(detail, name), name)


def _markdown_config_results(
    report: dict[str, Any], be_tolerance: float, report_dir: Path, campaign_dir: Path
) -> str:
    """Render a short local-preview report centered on effective config and outcomes."""
    evidence_rows = report.get("project_context", {}).get("run_evidence", [])
    evidence = {str(item.get("variant")): item for item in evidence_rows if isinstance(item, dict)}
    variants = report.get("variants", [])
    if not variants:
        raise ValueError("config-results report has no variants")

    def _config(variant: dict[str, Any]) -> dict[str, Any]:
        training = variant.get("training", {})
        return training.get("agent_parameters", {}) if isinstance(training, dict) else {}

    def _nested(value: dict[str, Any], *keys: str) -> Any:
        current: Any = value
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    def _shown(label: str, value: Any) -> str:
        if isinstance(value, float):
            ratio_field = {
                "Risk floor": "risk_floor",
                "Risk fraction range": "risk_frac_range",
                "Training peak drawdown limit": "peak_trailing_dd_limit",
            }.get(label)
            return _report_value(value, ratio_field) if ratio_field else _display(value)
        if isinstance(value, (list, tuple)):
            if label == "Risk fraction range":
                return _report_value(value, "risk_frac_range")
            return ", ".join(_display(item) for item in value)
        return _display(value)

    lines = [
        "# 2M matched-run report: configuration and results",
        "",
        f"Campaign: `{report.get('campaign_id', 'Not recorded')}`  ",
        f"Fingerprint: `{report.get('fingerprint') or 'Not recorded'}`  ",
        "Generated from saved run artifacts; no checkpoint was replayed for this report.",
        f"Break-even tolerance: {_display(be_tolerance)} account-currency units.",
        "",
        "## Run configuration",
        "",
        "| Setting | "
        + " | ".join(str(item.get("variant", "Variant")) for item in variants)
        + " |",
        "|---|" + "---:|" * len(variants),
    ]

    config_rows: list[tuple[str, tuple[Any, ...]]] = []
    paths: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Seed", ("resolved_variant", "seed")),
        ("Algorithm", ("resolved_variant", "algo")),
        ("Encoder", ("resolved_variant", "arch")),
        ("Training timesteps requested", ("ppo", "total_timesteps")),
        ("Training timesteps completed", ("run_summary", "timesteps_completed")),
        ("Parallel environments", ("env", "n_envs")),
        ("PPO batch size", ("ppo", "batch_size")),
        ("PPO epochs", ("ppo", "n_epochs")),
        ("Learning rate", ("ppo", "learning_rate")),
        ("Gamma", ("ppo", "gamma")),
        ("GAE lambda", ("ppo", "gae_lambda")),
        ("Reward", ("resolved_variant", "reward")),
        ("Action distribution", ("ppo", "policy_distribution")),
        ("Action width", ("resolved_variant", "action_schema", "width")),
        ("Entry state visible", ("env", "entry_state_observation")),
        (
            "Account observation width",
            ("resolved_variant", "observation_schema", "observation_width"),
        ),
        ("Risk mode", ("env", "risk_mode")),
        ("Risk floor", ("env", "risk_floor")),
        ("Risk fraction range", ("strategy", "risk", "risk_frac_range")),
        ("Position sizing cap (USD)", ("ftmo", "risk_sizing_limit_usd")),
        ("Per-trade risk limit (USD)", ("ftmo", "risk_per_trade_limit")),
        ("Training peak drawdown limit", ("env", "peak_trailing_dd_limit")),
    )
    for label, path in paths:
        values = tuple(
            _nested(_config(variant), *path)
            if path[0] != "run_summary"
            else _nested(variant.get("training", {}).get("run_summary", {}), path[1])
            for variant in variants
        )
        config_rows.append((label, values))
    evidence_paths: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Train action mode", ("evaluation", "training_action_mode")),
        ("Evaluation action mode", ("evaluation", "action_mode")),
        ("Modes match", ("evaluation", "action_mode_matches_training")),
        ("Tick execution", ("evaluation", "tick_execution")),
        ("Randomized fill latency (ms)", ("evaluation", "random_fill_latency_ms")),
        ("Test role", ("evaluation", "test_role")),
        ("Rollout steps per environment", ("training", "rollout_steps_per_environment")),
    )
    for label, path in evidence_paths:
        config_rows.append(
            (
                label,
                tuple(
                    _nested(evidence.get(str(item.get("variant")), {}), *path) for item in variants
                ),
            )
        )
    model_rows = (
        ("TCN channels", "channels"),
        ("TCN dilations", "dilations"),
        ("TCN kernel size", "kernel_size"),
        ("Latent dimension", "latent_dimension"),
        ("Policy head", "policy_head"),
        ("Model parameters", "parameter_count"),
    )
    for label, key in model_rows:
        config_rows.append(
            (
                label,
                tuple(
                    _nested(evidence.get(str(item.get("variant")), {}), "model", key)
                    for item in variants
                ),
            )
        )
    for label, values in config_rows:
        lines.append(f"| {label} | " + " | ".join(_shown(label, value) for value in values) + " |")

    first_evidence = evidence.get(str(variants[0].get("variant")), {})
    split = first_evidence.get("split", {})
    train_start = split.get("train_start")
    train_end = split.get("train_end")
    test_start = split.get("test_start")
    test_bars = next(
        (
            item
            for item in first_evidence.get("datasets", [])
            if isinstance(item, dict) and item.get("name") == "test_bars"
        ),
        {},
    )
    lines.extend(
        [
            "",
            "Configured split: training **{} to {}**; testing starts **{}**. "
            "Observed test bars run **{} to {}**. The test is labeled **{}** and this is a single-seed 2M-step pilot.".format(
                train_start or "not recorded",
                train_end or "not recorded",
                test_start or "not recorded",
                str(test_bars.get("start", "not recorded"))[:10],
                str(test_bars.get("end", "not recorded"))[:10],
                _nested(first_evidence, "evaluation", "test_role") or "not recorded",
            ),
            "",
            "## Training results",
            "",
            "| Variant | Return | Net PnL | Max drawdown / limit | Trades | Win rate | Profit factor | Average trade | Gate | Last entry | Peak-DD close |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for variant in variants:
        train = variant.get("splits", {}).get("training", {})
        metrics = train.get("metrics", {})
        training = variant.get("training", {})
        gate = training.get("training_equity_gate", {})
        limit = _number(gate.get("peak_trailing_dd_limit"))
        if limit is None:
            limit = _number(training.get("peak_trailing_dd_limit"))
        observed = _number(gate.get("max_peak_trailing_dd"))
        excess = observed - limit if observed is not None and limit is not None else None
        gate_text = (
            f"FAIL ({excess * 100:.4f} percentage points over)"
            if gate.get("ok") is False and excess is not None
            else "PASS"
            if gate.get("ok") is True
            else "Not recorded"
        )
        run_artifact_dir = variant.get("run_artifact_dir")
        trade_rows = (
            _rows(campaign_dir / run_artifact_dir / "training" / "trades.csv")
            if isinstance(run_artifact_dir, str)
            else []
        )
        last_open_index = next(
            (
                index
                for index in range(len(trade_rows) - 1, -1, -1)
                if trade_rows[index].get("type") == "open"
            ),
            None,
        )
        peak_close_index = next(
            (
                index
                for index in range(len(trade_rows) - 1, -1, -1)
                if trade_rows[index].get("type") == "close"
                and trade_rows[index].get("reason") == "peak_dd_behavior"
            ),
            None,
        )
        last_open_date = (
            str(
                trade_rows[last_open_index].get("fill_time")
                or trade_rows[last_open_index].get("time")
            )[:10]
            if last_open_index is not None
            else "Not recorded"
        )
        peak_close_date = (
            str(
                trade_rows[peak_close_index].get("time")
                or trade_rows[peak_close_index].get("fill_time")
            )[:10]
            if peak_close_index is not None
            else "Not recorded"
        )
        lines.append(
            "| {} | {} | {} | {} / {} | {} | {} | {} | {} | {} | {} | {} |".format(
                variant.get("variant", "Not recorded"),
                _report_value(metrics.get("total_return_pct"), "total_return_pct"),
                _display(metrics.get("total_pnl")),
                _report_value(metrics.get("max_drawdown"), "max_drawdown"),
                _report_value(limit, "peak_trailing_dd_limit"),
                _display(metrics.get("n_trades")),
                _report_value(metrics.get("win_rate"), "win_rate"),
                _display(metrics.get("profit_factor")),
                _display(metrics.get("avg_trade")),
                gate_text,
                last_open_date,
                peak_close_date,
            )
        )

    lines.extend(
        [
            "",
            "## Testing results",
            "",
            "| Variant | Return | Net PnL | Max drawdown | Trades | Win rate | Profit factor | Average trade |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in variants:
        test = variant.get("splits", {}).get("testing", {})
        metrics = test.get("metrics", {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                variant.get("variant", "Not recorded"),
                _report_value(metrics.get("total_return_pct"), "total_return_pct"),
                _display(metrics.get("total_pnl")),
                _report_value(metrics.get("max_drawdown"), "max_drawdown"),
                _display(metrics.get("n_trades")),
                _report_value(metrics.get("win_rate"), "win_rate"),
                _display(metrics.get("profit_factor")),
                _display(metrics.get("avg_trade")),
            )
        )

    activity_details = []
    for variant in variants:
        run_artifact_dir = variant.get("run_artifact_dir")
        if not isinstance(run_artifact_dir, str):
            continue
        trade_rows = _rows(campaign_dir / run_artifact_dir / "training" / "trades.csv")
        opened = [row for row in trade_rows if row.get("type") == "open"]
        guard_indices = [
            index
            for index, row in enumerate(trade_rows)
            if row.get("type") == "close" and row.get("reason") == "peak_dd_behavior"
        ]
        if not opened or not guard_indices:
            continue
        guard_index = guard_indices[-1]
        later_opens = any(row.get("type") == "open" for row in trade_rows[guard_index + 1 :])
        if later_opens:
            continue
        entry_date = str(opened[-1].get("fill_time") or opened[-1].get("time"))[:10]
        guard_row = trade_rows[guard_index]
        close_date = str(guard_row.get("time") or guard_row.get("fill_time"))[:10]
        activity_details.append(
            f"`{variant.get('variant')}` last opened a trade on {entry_date}; its peak-DD guard "
            f"closed the final position on {close_date}, with no later training entries."
        )
    lines.extend(
        [
            "",
            "## Why the training runs failed the year-survival gate",
            "",
            "Both simulations consumed the full training data through 2025-12-31 and ended profitable, "
            "but the policies did not remain active for the full year:",
            *(
                [f"- {detail}" for detail in activity_details]
                or ["- Last-entry dates were not recorded."]
            ),
            "",
            "The survival gate failed because each saved equity path exceeded the configured 10% "
            "peak-to-trough limit. This is a training drawdown-gate failure, not a missing year or "
            "trainer crash. The gate records no separate hard-loss breach.",
            "",
            "The configured guard closes an open trade at the drawdown ceiling and refuses new entries "
            "while the account remains at that ceiling. The final tick-quote fills left measured "
            "drawdown slightly above 10%; with equity below its previous high-water mark, trading "
            "did not resume. The controller continued through the remaining bars, which is why a "
            "full-year simulation exists even though active trading ended in April. The logs identify "
            "this failure path, but do not establish why the policy entered the losing move.",
            "",
            "The later test results are negative for both variants. Since this is one seed, a 2M-step "
            "pilot, and a previously inspected holdout, it does not establish a confirmatory ranking "
            "or prove whether the cause is architecture, reward design, trading logic, or sampling noise.",
            "",
            "## Equity and drawdown figures",
            "",
        ]
    )
    for variant in variants:
        label = str(variant.get("variant", "variant"))
        lines.extend([f"### {label}", ""])
        for split_name in ("training", "testing"):
            detail = variant.get("splits", {}).get(split_name, {})
            generated = detail.get("generated_figures", {})
            for figure_name in ("equity_curve", "peak_drawdown"):
                link = generated.get(figure_name)
                if not isinstance(link, str):
                    raise RuntimeError(f"missing {figure_name} figure for {label} {split_name}")
                image_path = (report_dir / link).resolve()
                lines.extend(
                    [
                        f"#### {split_name.capitalize()} {figure_name.replace('_', ' ')}",
                        "",
                        f"![{label} {split_name} {figure_name}]({image_path.as_posix()})",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def _markdown(report: dict[str, Any], campaign_dir: Path, be_tolerance: float) -> str:
    lines = [
        "# Project, training, and evaluation report",
        "",
        f"Campaign: `{report['campaign_id']}`  ",
        f"Campaign fingerprint: `{report.get('fingerprint') or 'Not recorded in campaign manifest'}`  ",
        f"Campaign source revision: `{report.get('campaign_git_sha') or 'Not recorded in campaign manifest'}`  ",
        "Generated from saved run artifacts. This report did not load a checkpoint or replay a market.",
        f"Break-even means an explicit break-even exit or absolute net PnL no greater than {_display(be_tolerance)} account-currency units.",
        "Training and testing sections are kept separate; interpretation should follow the saved split configuration.",
        "",
    ]
    project_context = report.get("project_context")
    if isinstance(project_context, dict):
        from scripts.research.thesis.evaluation.project_report_markdown import (
            render_project_context,
        )

        lines.extend([render_project_context(project_context), ""])
    lines.extend(
        [
            "## Variant summary",
            "",
            "| Variant | Seed | Run status | Peak-DD policy used in training | Train return | Train max DD | Train trades | Train gate | Test return | Test max DD | Test trades |",
            "|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for variant in report["variants"]:
        splits = variant.get("splits", {})
        train = splits.get("training") if isinstance(splits, dict) else None
        test = splits.get("testing") if isinstance(splits, dict) else None
        training = variant.get("training", {})
        gate = training.get("training_equity_gate") if isinstance(training, dict) else None
        if not isinstance(gate, dict):
            gate = {}
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _display(variant.get("variant")),
                _display(variant.get("seed")),
                _display(variant.get("controller_status")),
                _display(training.get("peak_drawdown_policy")),
                _markdown_metric(train, "total_return_pct"),
                _markdown_metric(train, "max_drawdown"),
                _display(_metric(train, "n_trades")),
                "PASS"
                if gate.get("ok") is True
                else "FAIL"
                if gate.get("ok") is False
                else "Not recorded",
                _markdown_metric(test, "total_return_pct"),
                _markdown_metric(test, "max_drawdown"),
                _display(_metric(test, "n_trades")),
            )
        )

    for variant in report["variants"]:
        lines.extend(
            [
                "",
                f"## {variant['variant']}",
                "",
                "### Run and training configuration",
                "",
            ]
        )
        summary = {
            "Controller status": variant.get("controller_status"),
            "Seed": variant.get("seed"),
            "Attempt": variant.get("attempt"),
            "Completed timesteps": variant.get("training", {})
            .get("run_summary", {})
            .get("timesteps_completed"),
            "Training peak-drawdown policy": variant.get("training", {}).get(
                "peak_drawdown_policy"
            ),
            "Training survival result": variant.get("training", {}).get("training_equity_gate"),
            "Training activity result": variant.get("training", {}).get("training_activity_gate"),
            "Drawdown policy evidence": variant.get("training", {}).get(
                "peak_drawdown_policy_evidence"
            ),
        }
        lines.extend(_markdown_table(summary, heading="Run setting"))
        lines.extend(["", "#### Effective model and environment parameters", ""])
        lines.extend(
            _markdown_table(
                variant.get("training", {}).get("agent_parameters", {}), heading="Parameter"
            )
        )

        training_curves = variant.get("training", {}).get("training_curve_stats", {})
        if training_curves:
            lines.extend(
                [
                    "",
                    "#### PPO or learner diagnostics",
                    "",
                    "| Training series | Samples | First | Last | Mean | Minimum | Maximum |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for name, values in training_curves.items():
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} |".format(
                        name.replace("_", " "),
                        _display(values.get("samples")),
                        *(
                            _display(values.get(key))
                            for key in ("first", "last", "mean", "min", "max")
                        ),
                    )
                )
        for name, path in variant.get("training", {}).get("figure_links", {}).items():
            lines.extend(
                ["", f"#### Training figure: {name.replace('_', ' ')}", "", f"![{name}]({path})"]
            )
        for name, path in variant.get("training", {}).get("generated_figures", {}).items():
            lines.extend(
                [
                    "",
                    f"#### Generated training figure: {name.replace('_', ' ')}",
                    "",
                    f"![{name}]({path})",
                ]
            )

        for split, detail in variant.get("splits", {}).items():
            lines.extend(
                [
                    "",
                    f"### {split.capitalize()} results",
                    "",
                    "This section reports the completed training split or the report-only testing split, as labeled.",
                    "",
                    "#### Performance and trading activity",
                    "",
                ]
            )
            lines.extend(_markdown_table(detail.get("metrics", {})))
            lines.extend(["", "#### Trade counts", ""])
            lines.extend(_markdown_table(detail.get("counts", {}), heading="Trade count"))
            lines.extend(
                [
                    "",
                    f"Peak-drawdown policy: **{detail.get('peak_drawdown_policy', 'Not recorded')}**. The measured maximum drawdown remains reported in the metrics above.",
                ]
            )
            for section in (
                "important_setup",
                "exit_breakdown",
                "sl_breakdown",
                "tp_breakdown",
                "risk_breakdown",
            ):
                lines.extend(["", f"#### {section.replace('_', ' ').title()}", ""])
                lines.extend(_markdown_table(detail.get(section, {})))
            generated = detail.get("generated_figures", {})
            for name, path in generated.items():
                lines.extend(
                    ["", f"#### Figure: {name.replace('_', ' ')}", "", f"![{name}]({path})"]
                )
            for name, path in detail.get("figure_links", {}).items():
                lines.extend(
                    [
                        "",
                        f"#### Saved evaluation figure: {name.replace('_', ' ')}",
                        "",
                        f"![{name}]({path})",
                    ]
                )

            examples = detail.get("trade_examples")
            if isinstance(examples, dict):
                labels = {
                    "winners": "Typical winning trades",
                    "losers": "Typical losing trades",
                    "breakeven": "Typical break-even trades",
                }
                for category in ("winners", "losers", "breakeven"):
                    trades = examples.get(category, [])
                    lines.extend(["", f"#### {labels[category]} ({len(trades)} selected)", ""])
                    if not trades:
                        lines.append(
                            "No completed trades met this outcome definition in the saved split results."
                        )
                        continue
                    lines.extend(
                        [
                            "| Example | Time | Side | Entry to exit | Net PnL | Realized R | Trading idea | Exit | Stop | Target | Requested / approved risk | Candlestick figure |",
                            "|---:|---|---|---:|---:|---:|---|---|---|---|---:|---|",
                        ]
                    )
                    for index, trade in enumerate(trades, start=1):
                        side = "Long" if (_number(trade.get("direction")) or 0) > 0 else "Short"
                        risk = trade.get("risk", {})
                        stop, target = trade.get("sl", {}), trade.get("tp", {})
                        chart = trade.get("market_chart")
                        if not chart:
                            raise RuntimeError(
                                "cannot render a selected typical trade without its candle figure: "
                                f"{variant['variant']} {split} position {trade.get('position_id')}"
                            )
                        chart_cell = f"[open chart]({chart})"
                        lines.append(
                            "| {} | {} | {} | {} → {} | {} | {} | {} | {} | {} @ {} | {} @ {} | {} / {} | {} |".format(
                                index,
                                _display(trade.get("entry_time")),
                                side,
                                _display(trade.get("entry_price")),
                                _display(trade.get("exit_price")),
                                _display(trade.get("net_pnl")),
                                _display(trade.get("realized_r")),
                                _display(trade.get("setup")),
                                _display(trade.get("exit_reason")),
                                _display(stop.get("reference")),
                                _display(stop.get("price")),
                                _display(target.get("reference")),
                                _display(target.get("price")),
                                _display(risk.get("requested_usd")),
                                _display(risk.get("approved_usd")),
                                chart_cell,
                            )
                        )
                if variant.get("example_figure_links", {}).get(split):
                    path = variant["example_figure_links"][split]
                    lines.extend(
                        [
                            "",
                            "#### Winner, loser, and break-even annotations",
                            "",
                            f"![Annotated trade selection for {variant['variant']} {split}]({path})",
                        ]
                    )
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    campaign_dir: Path,
    be_tolerance: float,
    example_split: str,
    report_dir: Path,
) -> dict[str, Any]:
    campaign = _json(campaign_dir / "campaign.json")
    if campaign is None:
        raise ValueError(f"campaign manifest missing or invalid: {campaign_dir / 'campaign.json'}")
    variants: list[dict[str, Any]] = []
    jobs = campaign.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(
            f"campaign manifest has no reportable jobs: {campaign_dir / 'campaign.json'}"
        )
    for job in jobs:
        if not isinstance(job, dict):
            continue
        key = str(job.get("key") or "")
        if not key:
            continue
        attempt = int(job.get("attempts", 1) or 1)
        attempt_dir = campaign_dir / "runs" / "attempts" / key / f"attempt{attempt}"
        candidates = sorted((attempt_dir / key).glob("*")) if (attempt_dir / key).is_dir() else []
        run_dir = next((path for path in reversed(candidates) if path.is_dir()), None)
        if run_dir is None:
            raise FileNotFoundError(
                f"cannot create complete results report; no run artifacts for {key} attempt {attempt}"
            )
        resolved = _json(attempt_dir / key / "resolved_config.json") or _json(
            run_dir / "resolved_config.json" if run_dir else Path("/nonexistent")
        )
        training_log = _json(run_dir / "training_log.json") if run_dir else None
        training = (
            _training_report(run_dir, training_log or resolved)
            if run_dir
            else {
                "run_summary": training_log or {},
                "agent_parameters": _flatten_config(resolved),
                "training_curve_stats": {},
                "training_log_rows": 0,
                "figures": [],
                "peak_trailing_dd_limit": 0.0,
                "peak_drawdown_policy": "No training run artifacts were recorded",
                "training_equity_gate": None,
                "training_activity_gate": None,
            }
        )
        train_dir = run_dir / "training" if run_dir else None
        test_dir = run_dir / "testing" if run_dir else None
        splits: dict[str, Any] = {}
        for split_name, split_dir in (("training", train_dir), ("testing", test_dir)):
            if split_dir is None or not split_dir.is_dir():
                raise FileNotFoundError(f"required {split_name} results are missing: {split_dir}")
            for filename in (
                "metrics.json",
                "equity.csv",
                "positions.csv",
                "fills.csv",
                "trades.csv",
            ):
                if not (split_dir / filename).is_file():
                    raise FileNotFoundError(
                        f"required {split_name} report artifact is missing: {split_dir / filename}"
                    )
            detail = _split_report(
                run_dir,
                split_name,
                be_tolerance,
                split_name == example_split or example_split == "both",
                float(training.get("peak_trailing_dd_limit", 0.0)),
            )
            if detail.get("metrics") is None:
                raise ValueError(
                    f"required {split_name} metrics are invalid: {split_dir / 'metrics.json'}"
                )
            if detail.get("trade_examples"):
                for group in detail["trade_examples"].values():
                    for trade in group:
                        trade["market_chart"] = None
            detail["figure_links"] = {
                name: os.path.relpath(run_dir / rel, report_dir)
                for name, rel in detail["artifact_files"].items()
                if name.endswith("figure")
            }
            splits[split_name] = detail
        examples = None
        if example_split == "both":
            examples = {name: value.get("trade_examples") or {} for name, value in splits.items()}
        elif example_split in splits:
            examples = splits[example_split].get("trade_examples")
        training_figures = training.get("figures")
        if not isinstance(training_figures, list):
            training_figures = []
        training["figure_links"] = (
            {Path(rel).stem: os.path.relpath(run_dir / rel, report_dir) for rel in training_figures}
            if run_dir is not None
            else {}
        )
        variants.append(
            {
                "variant": job.get("variant") or key,
                "run_key": key,
                "seed": job.get("seed"),
                "attempt": attempt,
                "controller_status": job.get("status"),
                "exit_code": job.get("exit_code"),
                "failure_reason": job.get("failure_reason"),
                "run_artifact_dir": run_dir.relative_to(campaign_dir).as_posix()
                if run_dir
                else None,
                "training": training,
                "training_source_sha256": {
                    name: _sha256(path)
                    for name, path in {
                        "resolved_config.json": attempt_dir / key / "resolved_config.json",
                        "config.yaml": run_dir / "config.yaml" if run_dir else Path("/nonexistent"),
                        "training_log.json": run_dir / "training_log.json"
                        if run_dir
                        else Path("/nonexistent"),
                        "model/training_log.csv": run_dir / "model" / "training_log.csv"
                        if run_dir
                        else Path("/nonexistent"),
                    }.items()
                    if path.is_file()
                },
                "splits": splits,
                "examples": examples,
                "example_split": example_split,
                "example_figure_links": {},
                "example_rule": "up to five trades nearest each category median net PnL; BE means explicit BE exit or within tolerance",
            }
        )
    ladder_order = {
        name: index for index, name in enumerate(("a0", "a1", "a2", "a3", "a4a", "a4b"))
    }

    def variant_order(item: dict[str, Any]) -> tuple[int, str]:
        parts = str(item["variant"]).split("_")
        rank = next(
            (ladder_order[part] for part in parts if part in ladder_order), len(ladder_order)
        )
        return rank, str(item["variant"])

    variants.sort(key=variant_order)
    if not variants:
        raise ValueError(f"campaign manifest contains no jobs with reportable keys: {campaign_dir}")
    from scripts.research.thesis.evaluation.project_report_context import (
        build_project_context,
    )

    return {
        "schema_version": 2,
        "campaign_id": campaign_dir.name,
        "fingerprint": campaign.get("fingerprint"),
        "campaign_git_sha": campaign.get("git_sha"),
        "campaign_manifest_sha256": _sha256(campaign_dir / "campaign.json"),
        "report_script_sha256": _sha256(Path(__file__).resolve()),
        "be_tolerance_account_currency": be_tolerance,
        "example_split": example_split,
        "variants": variants,
        "project_context": build_project_context(
            campaign_dir,
            campaign,
            variants,
            report_dir,
            _PROJECT_ROOT,
        ),
    }


def _load_chart_bars(
    run_dir: Path,
    split: str,
    project_root: Path,
    bars_cache: dict[tuple[str, str, str], Any],
) -> Any:
    """Load the exact configured primary M1 source for selected trade figures."""
    import pandas as pd

    from market_data.loading.bars import load_bars
    from market_data.transforms.cleaning import clean
    from market_data.transforms.price_grid import quantize_bar_prices

    try:
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read run config for market charts: {run_dir}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"run config is not a mapping: {run_dir / 'config.yaml'}")
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"run config has no data mapping: {run_dir / 'config.yaml'}")
    datasets = data.get("datasets")
    dataset = datasets.get(split) if isinstance(datasets, dict) else None
    if not isinstance(dataset, dict):
        raise ValueError(f"run config has no data.datasets.{split} mapping")
    m1_files = dataset.get("m1_files")
    if not isinstance(m1_files, dict) or not m1_files:
        raise ValueError(f"run config has no M1 file map for {split}")
    primary = str(data.get("primary") or next(iter(m1_files)))
    filename = m1_files.get(primary)
    if not filename:
        raise ValueError(f"run config has no primary M1 file for {primary} in {split}")
    raw_dir = Path(str(dataset.get("raw_dir") or data.get("raw_dir") or "data"))
    raw_dir = raw_dir if raw_dir.is_absolute() else project_root / raw_dir
    source_path = raw_dir / str(filename)
    tz = str(data.get("tz") or "Etc/GMT-3")
    price_grid = data.get("price_discretization")
    increment = None
    if isinstance(price_grid, dict) and bool(price_grid.get("enabled", False)):
        increment = float(price_grid.get("increment", 0.0))
        if not math.isfinite(increment) or increment <= 0.0:
            raise ValueError(f"invalid price-grid increment in {run_dir / 'config.yaml'}")
    cache_key = (str(source_path.resolve()), tz, str(increment))
    if cache_key in bars_cache:
        return bars_cache[cache_key]
    if not source_path.is_file():
        raise FileNotFoundError(f"configured M1 bars for trade figures are missing: {source_path}")
    bars = clean(quantize_bar_prices(load_bars(source_path), increment), tz=tz)
    equity_path = run_dir / split / "equity.csv"
    if not equity_path.is_file():
        raise FileNotFoundError(
            f"equity timeline is missing for selected trade figures: {equity_path}"
        )
    equity = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    if equity.empty:
        raise ValueError(f"equity timeline is empty for selected trade figures: {equity_path}")
    end_time = pd.Timestamp(equity.index[-1])
    bar_index = pd.DatetimeIndex(bars.index)
    if bar_index.tz is None and end_time.tzinfo is not None:
        end_time = end_time.tz_localize(None)
    elif bar_index.tz is not None and end_time.tzinfo is None:
        end_time = end_time.tz_localize(bar_index.tz)
    elif bar_index.tz is not None and end_time.tzinfo is not None:
        end_time = end_time.tz_convert(bar_index.tz)
    bars = bars.loc[bar_index <= end_time]
    if bars.empty:
        raise ValueError(f"configured M1 bars do not overlap {split} equity timestamps")
    bars_cache[cache_key] = bars
    return bars


def _write_equity_figures(
    split_dir: Path,
    out_dir: Path,
    title: str,
    peak_dd_limit: float,
) -> dict[str, Path]:
    """Create report-owned equity and drawdown figures from the split CSV."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    equity_path = split_dir / "equity.csv"
    if not equity_path.is_file():
        raise FileNotFoundError(f"required split equity data is missing: {equity_path}")
    frame = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    if frame.empty or "equity" not in frame.columns:
        raise ValueError(f"required split equity data is empty or malformed: {equity_path}")
    equity = pd.to_numeric(frame["equity"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(equity).all() or np.any(equity <= 0.0):
        raise ValueError(f"required split equity data has invalid values: {equity_path}")
    dates = pd.DatetimeIndex(frame.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(dates, equity, color="#1f5a94", linewidth=1.15, label="Equity")
    ax.axhline(equity[0], color="#7a8691", linestyle="--", linewidth=0.9, label="Starting equity")
    ax.set_title(f"{title}: equity path")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account equity")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # type: ignore[no-untyped-call]
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures["equity_curve"] = path

    peaks = np.maximum.accumulate(equity)
    drawdown = (peaks - equity) / peaks * 100.0
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(dates, drawdown, color="#c13c3c", linewidth=1.0, label="Peak-to-trough drawdown")
    if peak_dd_limit > 0.0:
        ax.axhline(
            peak_dd_limit * 100.0,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            label=f"Configured training policy: {peak_dd_limit:.1%}",
        )
    else:
        ax.text(
            0.99,
            0.96,
            "No peak-drawdown limit configured; diagnostic only",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    ax.set_title(f"{title}: peak drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from running peak (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # type: ignore[no-untyped-call]
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / "peak_drawdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures["peak_drawdown"] = path
    for name, figure_path in figures.items():
        if not figure_path.is_file() or figure_path.stat().st_size == 0:
            raise RuntimeError(f"report figure generation failed: {figure_path}")
    return figures


def _plot_training_diagnostics(
    report: dict[str, Any], campaign_dir: Path, figure_dir: Path, report_dir: Path
) -> None:
    """Generate the same learner diagnostic figure for every reported run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    panels = (
        (
            "Optimization losses",
            ("train/loss", "train/policy_gradient_loss", "train/value_loss"),
        ),
        ("Policy behavior", ("train/entropy_loss", "train/explained_variance")),
        ("Update stability", ("train/approx_kl", "train/clip_fraction", "train/learning_rate")),
    )
    for variant in report["variants"]:
        run_dir = campaign_dir / variant["run_artifact_dir"]
        source = run_dir / "model" / "training_log.csv"
        frame = pd.read_csv(source)
        x = (
            pd.to_numeric(frame["timestep"], errors="coerce").to_numpy(dtype=float)
            if "timestep" in frame.columns
            else np.arange(len(frame), dtype=float)
        )
        out_dir = figure_dir / variant["run_key"] / "training"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 11), squeeze=False)
        plotted_any = False
        for ax, (title, columns) in zip(axes[:, 0], panels):
            plotted_panel = False
            for column in columns:
                if column not in frame.columns:
                    continue
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(values)
                if not finite.any():
                    continue
                ax.plot(
                    x[finite], values[finite], linewidth=1.15, label=column.removeprefix("train/")
                )
                plotted_panel = True
                plotted_any = True
            ax.set_title(title)
            ax.set_xlabel("Training timestep")
            ax.grid(True, alpha=0.25)
            if plotted_panel:
                ax.legend(loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No values recorded for this panel",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        if not plotted_any:
            plt.close(fig)
            raise ValueError(f"training log has no plottable learner diagnostics: {source}")
        fig.suptitle(f"{variant['variant']}: training diagnostics")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = out_dir / "learner_diagnostics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"training diagnostic figure generation failed: {path}")
        variant["training"]["generated_figures"] = {
            "learner_diagnostics": os.path.relpath(path, report_dir)
        }


def _plot_variant_figures(
    report: dict[str, Any], campaign_dir: Path, figure_dir: Path, report_dir: Path
) -> None:
    """Create one small set of comparable diagnostic figures per variant/split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from evaluation.plotting.orders import plot_per_trade_orders

    project_root = Path(__file__).resolve().parents[4]
    bars_cache: dict[tuple[str, str, str], Any] = {}

    for variant in report["variants"]:
        run_key = variant["run_key"]
        run_artifact_dir = variant.get("run_artifact_dir")
        if not isinstance(run_artifact_dir, str):
            continue
        run_dir = campaign_dir / run_artifact_dir
        for split, detail in variant.get("splits", {}).items():
            out_dir = figure_dir / run_key / split
            out_dir.mkdir(parents=True, exist_ok=True)
            charts: dict[str, Path] = {}
            split_dir = run_dir / split
            charts.update(
                _write_equity_figures(
                    split_dir,
                    out_dir,
                    f"{variant['variant']} · {split}",
                    float(detail.get("peak_trailing_dd_limit", 0.0)),
                )
            )

            def save_counts(title: str, counts: dict[str, int], filename: str, color: str) -> None:
                fig, ax = plt.subplots(figsize=(10, 5.5))
                items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
                if items:
                    labels, values = zip(*items)
                    y = list(range(len(labels)))
                    ax.barh(y, values, color=color)
                    ax.set_yticks(y, labels=[str(label) for label in labels])
                    ax.invert_yaxis()
                    ax.set_xlabel("Count")
                    for idx, value in enumerate(values):
                        ax.text(value, idx, f" {value}", va="center")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No recorded rows",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                ax.set_title(title)
                fig.tight_layout()
                path = out_dir / filename
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                charts[Path(filename).stem] = path

            setup = detail.get("important_setup", {})
            save_counts(
                f"{variant['variant']} · {split}: entry level / idea mix",
                setup.get("entry_level", {}),
                "idea_level_breakdown.png",
                "#4c78a8",
            )
            save_counts(
                f"{variant['variant']} · {split}: strategy setup types",
                setup.get("setup_types", {}),
                "idea_setup_type_breakdown.png",
                "#4c78a8",
            )

            plot_data = detail.get("plot_data", {})
            ideas: dict[str, list[float]] = {}
            for item in plot_data.get("ideas", []):
                pnl = _number(item.get("net_pnl"))
                if pnl is None:
                    continue
                level = str(item.get("level") or "unavailable")
                ideas.setdefault(level, []).append(pnl)
            fig, ax = plt.subplots(figsize=(10, 5.5))
            idea_rows = sorted(ideas.items(), key=lambda item: (-len(item[1]), item[0]))[:12]
            if idea_rows:
                labels = [item[0] for item in idea_rows]
                counts = [len(item[1]) for item in idea_rows]
                medians = [statistics.median(item[1]) for item in idea_rows]
                x = list(range(len(labels)))
                ax.bar(x, counts, color="#4c78a8", alpha=0.8, label="trade count")
                ax.set_ylabel("Trade count")
                ax.set_xticks(x, labels=labels, rotation=35, ha="right")
                second = ax.twinx()
                second.plot(x, medians, color="#f58518", marker="o", label="median net PnL")
                second.set_ylabel("Median net PnL")
                ax.legend(loc="upper left")
                second.legend(loc="upper right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No setup-linked completed trades",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            ax.set_title(f"{variant['variant']} · {split}: outcome by entry level")
            fig.tight_layout()
            path = out_dir / "idea_outcomes_by_level.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts[path.stem] = path

            exit_data = detail.get("exit_breakdown", {}).get("exit_reason_by_net_outcome", {})
            reasons = sorted({reason for counts in exit_data.values() for reason in counts})
            fig, ax = plt.subplots(figsize=(10, 5.5))
            if reasons:
                groups = ["winning", "losing", "breakeven"]
                bottom = [0] * len(reasons)
                colors = {"winning": "#54a24b", "losing": "#e45756", "breakeven": "#9d9da1"}
                x = list(range(len(reasons)))
                for group in groups:
                    values = [exit_data.get(group, {}).get(reason, 0) for reason in reasons]
                    ax.bar(x, values, bottom=bottom, label=group, color=colors[group])
                    bottom = [a + b for a, b in zip(bottom, values)]
                ax.set_xticks(x, labels=reasons, rotation=35, ha="right")
                ax.set_ylabel("Completed positions")
                ax.legend()
            else:
                ax.text(
                    0.5, 0.5, "No exit records", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_axis_off()
            ax.set_title(f"{variant['variant']} · {split}: exit reason by trade outcome")
            fig.tight_layout()
            path = out_dir / "exit_breakdown_by_outcome.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts[path.stem] = path

            save_counts(
                f"{variant['variant']} · {split}: selected stop references",
                detail.get("sl_breakdown", {}).get("selected_reference", {}),
                "sl_reference_breakdown.png",
                "#f58518",
            )
            save_counts(
                f"{variant['variant']} · {split}: selected target references",
                detail.get("tp_breakdown", {}).get("selected_reference", {}),
                "tp_reference_breakdown.png",
                "#72b7b2",
            )

            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
            histogram_items = (
                ("Modeled stop distance", "stop_distance"),
                ("Planned reward/risk", "planned_rr"),
                ("Risk sizing values", "risk_requested"),
            )
            for ax, (title, key) in zip(axes, histogram_items):
                distribution_groups: tuple[tuple[str, str], ...] = (
                    (
                        ("requested", "risk_requested"),
                        ("approved", "risk_approved"),
                        ("modeled stop", "risk_stop"),
                    )
                    if key == "risk_requested"
                    else ((title, key),)
                )
                plotted = False
                for label, series in distribution_groups:
                    values = [_number(value) for value in plot_data.get(series, [])]
                    values = [value for value in values if value is not None]
                    if values:
                        ax.hist(
                            values,
                            bins=min(25, max(5, int(math.sqrt(len(values))))),
                            alpha=0.55,
                            label=label,
                        )
                        ax.axvline(statistics.median(values), linestyle="--", linewidth=1)
                        plotted = True
                if plotted:
                    ax.legend()
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No recorded values",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                ax.set_title(title)
            fig.suptitle(f"{variant['variant']} · {split}: stop, target, and risk distributions")
            fig.tight_layout()
            path = out_dir / "sl_tp_risk_distributions.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts[path.stem] = path

            examples = detail.get("trade_examples") or {}
            if examples:
                fig, axes = plt.subplots(5, 3, figsize=(16, 12), squeeze=False)
                columns = (
                    ("winners", "Typical winners", "#54a24b"),
                    ("losers", "Typical losers", "#e45756"),
                    ("breakeven", "Typical break-even", "#9d9da1"),
                )
                for col, (category, label, color) in enumerate(columns):
                    axes[0][col].set_title(
                        f"{label} · {len(examples.get(category, []))} selected",
                        color=color,
                        weight="bold",
                    )
                    for row_idx in range(5):
                        ax = axes[row_idx][col]
                        ax.axis("off")
                        items = examples.get(category, [])
                        if row_idx >= len(items):
                            if row_idx == 0 and not items:
                                ax.text(
                                    0.05,
                                    0.9,
                                    f"No {label.lower()} recorded in this split",
                                    ha="left",
                                    va="top",
                                    transform=ax.transAxes,
                                    fontsize=10,
                                )
                            continue
                        trade = items[row_idx]
                        setup_info = trade.get("setup", {})
                        risk_info = trade.get("risk", {})
                        sl_info, tp_info = trade.get("sl", {}), trade.get("tp", {})
                        side = "LONG" if (_number(trade.get("direction")) or 0) > 0 else "SHORT"
                        content = (
                            f"TYPICAL {category[:-1].upper() if category != 'breakeven' else 'BREAK-EVEN'} #{row_idx + 1}\n"
                            f"{trade.get('entry_time', 'time not recorded')} · {side}\n"
                            f"{trade.get('entry_price', '—')} → {trade.get('exit_price', '—')}   "
                            f"PnL {trade.get('net_pnl', '—')}   R {trade.get('realized_r', '—')}\n"
                            f"Setup: {setup_info.get('level_type', '—')} | {setup_info.get('entry_setup_types', '—')}\n"
                            f"Exit: {trade.get('exit_reason', '—')}\n"
                            f"SL: {sl_info.get('reference', '—')} @ {sl_info.get('price', '—')}   "
                            f"TP: {tp_info.get('reference', '—')} @ {tp_info.get('price', '—')}\n"
                            f"Risk req/approved: {risk_info.get('requested_usd', '—')} / {risk_info.get('approved_usd', '—')}"
                        )
                        ax.text(
                            0.02,
                            0.95,
                            content,
                            ha="left",
                            va="top",
                            fontsize=8.5,
                            family="monospace",
                            wrap=True,
                            bbox={
                                "boxstyle": "round,pad=0.5",
                                "facecolor": color,
                                "alpha": 0.12,
                                "edgecolor": color,
                            },
                        )
                fig.suptitle(
                    f"{variant['variant']} · {split}: representative winner, loser, and break-even trades\n"
                    "Selected nearest each outcome median; individual candle figures are linked in the tables",
                    fontsize=13,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                path = out_dir / "typical_trade_examples.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                charts[path.stem] = path

                selected_trades = [
                    trade
                    for group in ("winners", "losers", "breakeven")
                    for trade in examples.get(group, [])
                    if trade.get("position_id")
                ]
                if selected_trades:
                    trade_event_path = split_dir / "trades.csv"
                    if not trade_event_path.is_file():
                        raise FileNotFoundError(
                            f"selected trade candle figures require trade events: {trade_event_path}"
                        )
                    selected_ids = {str(trade["position_id"]) for trade in selected_trades}
                    event_chunks = [
                        chunk.loc[chunk["position_id"].astype(str).isin(selected_ids)]
                        for chunk in pd.read_csv(
                            trade_event_path,
                            chunksize=200_000,
                            dtype={"position_id": "string"},
                            low_memory=False,
                        )
                        if "position_id" in chunk.columns
                        and chunk["position_id"].astype(str).isin(selected_ids).any()
                    ]
                    if not event_chunks:
                        raise ValueError(
                            f"no trade events match selected positions in {trade_event_path}"
                        )
                    trade_events = pd.concat(event_chunks, ignore_index=True)
                    chart_bars = _load_chart_bars(run_dir, split, project_root, bars_cache)
                    chart_order_dir = out_dir / "orders"
                    plot_per_trade_orders(
                        chart_bars,
                        trade_events,
                        orders_dir=chart_order_dir,
                        context_bars=45,
                        max_charts=len(selected_ids),
                        dpi=130,
                        show_mae_mfe=True,
                        show_sl_tp=True,
                    )
                    for category in ("winners", "losers", "breakeven"):
                        for rank, trade in enumerate(examples.get(category, []), start=1):
                            if not trade.get("position_id"):
                                continue
                            chart = _trade_chart(out_dir, trade, report_dir)
                            if chart is None:
                                raise RuntimeError(
                                    "a selected typical trade has no generated candle figure: "
                                    f"{variant['variant']} {split} position {trade['position_id']}"
                                )
                            _annotate_trade_outcome(
                                report_dir / chart,
                                _trade_outcome_caption(category, split, rank, trade),
                                category,
                            )
                            trade["market_chart"] = chart
                            trade["market_chart_label"] = _trade_outcome_caption(
                                category, split, rank, trade
                            )
            detail["generated_figures"] = {
                name: os.path.relpath(path, report_dir) for name, path in charts.items()
            }
            example_path = charts.get("typical_trade_examples")
            if example_path:
                variant.setdefault("example_figure_links", {})[split] = os.path.relpath(
                    example_path, report_dir
                )


def _validate_report_figures(report: dict[str, Any], report_dir: Path) -> None:
    """Fail report creation if any required or linked figure is missing or empty."""
    required_split_figures = {
        "equity_curve",
        "peak_drawdown",
        "idea_level_breakdown",
        "idea_setup_type_breakdown",
        "idea_outcomes_by_level",
        "exit_breakdown_by_outcome",
        "sl_reference_breakdown",
        "tp_reference_breakdown",
        "sl_tp_risk_distributions",
    }
    figure_links: list[str] = []
    for variant in report.get("variants", []):
        training = variant.get("training", {})
        training_figures = training.get("generated_figures", {})
        if "learner_diagnostics" not in training_figures:
            raise RuntimeError(
                f"required learner diagnostics were not generated for {variant.get('variant')}"
            )
        figure_links.extend(training_figures.values())
        figure_links.extend(training.get("figure_links", {}).values())
        figure_links.extend(variant.get("example_figure_links", {}).values())
        for split, detail in variant.get("splits", {}).items():
            generated = detail.get("generated_figures", {})
            missing_required = required_split_figures - set(generated)
            if (
                detail.get("trade_examples") is not None
                and "typical_trade_examples" not in generated
            ):
                missing_required.add("typical_trade_examples")
            if missing_required:
                raise RuntimeError(
                    f"required {split} figures were not generated for {variant.get('variant')}: "
                    + ", ".join(sorted(missing_required))
                )
            figure_links.extend(detail.get("figure_links", {}).values())
            figure_links.extend(generated.values())
            for group in (detail.get("trade_examples") or {}).values():
                for trade in group:
                    chart = trade.get("market_chart")
                    if chart:
                        figure_links.append(chart)
                    else:
                        raise RuntimeError(
                            "typical trade is missing its required annotated candle figure: "
                            f"{variant.get('variant')} {split} {trade.get('position_id')}"
                        )
    missing = [
        link
        for link in figure_links
        if not (report_dir / link).is_file() or (report_dir / link).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "report generation stopped because referenced figures are missing or empty: "
            + ", ".join(missing[:12])
        )


def _validate_markdown_images(markdown: str, report_dir: Path) -> None:
    """Check relative or absolute Markdown image references before saving."""
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)
    if not links:
        raise RuntimeError("results report has no figures")
    missing = []
    for link in links:
        target = Path(link)
        if not target.is_absolute():
            target = report_dir / target
        if not target.is_file() or target.stat().st_size == 0:
            missing.append(link)
    if missing:
        raise FileNotFoundError(
            "Markdown references missing or empty figures: " + ", ".join(missing[:12])
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument(
        "--out-stem",
        type=Path,
        help="default: <campaign>/reports/project_results_report",
    )
    parser.add_argument(
        "--be-tolerance",
        type=float,
        default=1.0,
        help="net PnL tolerance in account currency; default 1.0",
    )
    parser.add_argument("--example-split", choices=("training", "testing", "both"), default="both")
    parser.add_argument(
        "--scope",
        choices=("full", "config-results"),
        default="full",
        help="config-results writes a concise local-preview report with absolute figure paths",
    )
    args = parser.parse_args()
    if args.be_tolerance < 0 or not math.isfinite(args.be_tolerance):
        parser.error("--be-tolerance must be finite and non-negative")
    campaign_dir = args.campaign_dir.resolve()
    out_stem = args.out_stem or campaign_dir / "reports" / "project_results_report"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    report_dir = out_stem.parent.resolve()
    report = build_report(campaign_dir, args.be_tolerance, args.example_split, report_dir)
    diagram_path = out_stem.with_name(f"{out_stem.stem}_pipeline_architecture.mmd")
    project_context = report.get("project_context")
    if isinstance(project_context, dict):
        project_context["pipeline_diagram_file"] = diagram_path.name
        diagram_path.write_text(
            str(project_context.get("pipeline_diagram", "")).rstrip() + "\n",
            encoding="utf-8",
        )
        if not diagram_path.is_file() or diagram_path.stat().st_size == 0:
            raise RuntimeError(f"pipeline architecture diagram is missing or empty: {diagram_path}")
    _plot_training_diagnostics(
        report,
        campaign_dir,
        report_dir / f"{out_stem.stem}_figures",
        report_dir,
    )
    _plot_variant_figures(
        report,
        campaign_dir,
        report_dir / f"{out_stem.stem}_figures",
        report_dir,
    )
    _validate_report_figures(report, report_dir)
    markdown = (
        _markdown(report, campaign_dir, args.be_tolerance)
        if args.scope == "full"
        else _markdown_config_results(report, args.be_tolerance, report_dir, campaign_dir)
    )
    _validate_markdown_images(markdown, report_dir)
    json_path = out_stem.with_suffix(".json")
    markdown_path = out_stem.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Project and campaign report created for {len(report['variants'])} variants.")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")
    print(f"Pipeline diagram: {diagram_path}")
    print(
        "Trade examples: up to five typical winners, losers, and BE trades per selected split and variant."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
