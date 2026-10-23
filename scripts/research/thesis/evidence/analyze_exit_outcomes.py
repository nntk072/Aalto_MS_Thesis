#!/usr/bin/env python3
"""Build a provenance-aware exit/action analysis from campaign evaluation CSVs.

Example:
    python3 scripts/research/thesis/evidence/analyze_exit_outcomes.py \
      --campaign-root outputs/gpu_campaigns/<campaign> --split testing

The script only reads run manifests, positions.csv, and trades.csv. It does not
load checkpoints or rerun model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EMA_EXIT_THRESHOLD = 0.5
ACTION_EVENT_TYPES = {"open", "reject"}
TERMINAL_EVENT_TYPES = {"close", "stop_close", "tp_close", "eod_close", "forced_close"}
TP_EVENT_TYPES = {"tp_partial_close", "tp_close"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _completed(value: Any, exit_reason: Any) -> bool:
    if value is None or str(value).strip() == "":
        return bool(str(exit_reason or "").strip())
    return str(value).strip().lower() in {"true", "1", "yes"}


def _campaign_label(run_dir: Path, campaign_root: Path) -> str:
    try:
        parts = run_dir.relative_to(campaign_root).parts
    except ValueError:
        return run_dir.name
    if "attempts" in parts:
        index = parts.index("attempts")
        return "/".join(parts[index + 1 :])
    return "/".join(parts)


def _discover_runs(campaign_root: Path, split: str) -> list[dict[str, Any]]:
    attempts_root = campaign_root / "runs" / "attempts"
    search_root = attempts_root if attempts_root.is_dir() else campaign_root
    discovered: list[dict[str, Any]] = []
    for manifest_path in sorted(search_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        split_dir = run_dir / split
        positions_path = split_dir / "positions.csv"
        trades_path = split_dir / "trades.csv"
        if not positions_path.is_file() or not trades_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        discovered.append(
            {
                "run_dir": run_dir,
                "label": _campaign_label(run_dir, campaign_root),
                "manifest_path": manifest_path,
                "manifest": manifest,
                "positions_path": positions_path,
                "trades_path": trades_path,
                "positions": _read_csv(positions_path),
                "events": _read_csv(trades_path),
            }
        )
    if not discovered:
        raise FileNotFoundError(
            f"no run manifests with {split}/positions.csv and {split}/trades.csv found under {search_root}"
        )
    return discovered


def _terminal_records(
    positions: Sequence[dict[str, str]],
    events: Sequence[dict[str, str]],
    near_be_r: float,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, dict[str, str]]]]:
    terminal_by_position: dict[str, tuple[int, dict[str, str]]] = {}
    for index, event in enumerate(events):
        if event.get("type") not in TERMINAL_EVENT_TYPES:
            continue
        position_id = event.get("position_id", "")
        if position_id:
            terminal_by_position[position_id] = (index, event)

    result: list[dict[str, Any]] = []
    for position in positions:
        reason = position.get("exit_reason", "")
        if not _completed(position.get("completed"), reason):
            continue
        position_id = position.get("position_id", "")
        terminal = terminal_by_position.get(position_id)
        event_index = terminal[0] if terminal else None
        event = terminal[1] if terminal else {}
        pnl = _number(position.get("position_net_pnl"))
        if pnl is None:
            pnl = _number(position.get("net_pnl"))
        if pnl is None:
            continue
        realized_r = _number(event.get("position_realized_r"))
        if realized_r is None:
            realized_r = _number(event.get("realized_r"))
        if pnl > 0.0:
            outcome = "win"
        elif pnl < 0.0:
            outcome = "loss"
        else:
            outcome = "zero"
        near_be = realized_r is not None and abs(realized_r) <= near_be_r
        result.append(
            {
                "position_id": position_id,
                "exit_reason": reason or event.get("reason", "unknown"),
                "exit_event_type": event.get("type", ""),
                "pnl": pnl,
                "realized_r": realized_r,
                "outcome": outcome,
                "near_be": near_be,
                "terminal_event_index": event_index,
                "sl_ref": event.get("sl_ref", ""),
                "tp_ref": event.get("tp_ref", ""),
            }
        )
    return result, terminal_by_position


def _reference_families(reference: Any) -> list[str]:
    value = str(reference or "").lower()
    families: list[str] = []
    if "fvg" in value or "ifvg" in value:
        families.append("FVG/IFVG")
    if "swing" in value:
        families.append("swing")
    if any(token in value for token in ("asian", "london", "yesterday", "lastweek")):
        families.append("session/day/week level")
    if "dev_" in value:
        families.append("deviation")
    if not families:
        families.append("other")
    return families


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _action_summary(run: dict[str, Any], split_name: str) -> dict[str, Any]:
    manifest = run["manifest"]
    schema = manifest.get("action_schema", {})
    coordinates = schema.get("coordinates", []) if isinstance(schema, dict) else []
    action_rows = [
        event
        for event in run["events"]
        if event.get("type") in ACTION_EVENT_TYPES and _number(event.get("exit_u")) is not None
    ]
    exit_values = [
        value for event in action_rows if (value := _number(event.get("exit_u"))) is not None
    ]
    decoded = Counter(
        "ema_21" if value >= EMA_EXIT_THRESHOLD else "structural" for value in exit_values
    )
    logged_modes = Counter(
        event.get("exit_mode", "")
        for event in run["events"]
        if event.get("type") == "open" and event.get("exit_mode", "")
    )
    mismatches = 0
    for event in run["events"]:
        if event.get("type") != "open" or not event.get("exit_mode"):
            continue
        value = _number(event.get("exit_u"))
        if value is None:
            continue
        decoded_mode = "ema_21" if value >= EMA_EXIT_THRESHOLD else "structural"
        if decoded_mode != event.get("exit_mode"):
            mismatches += 1
    coordinate_rows: list[tuple[str, int, str, str]] = []
    for coordinate in coordinates:
        field = f"{coordinate}_u"
        values = [
            value for event in action_rows if (value := _number(event.get(field))) is not None
        ]
        if values:
            coordinate_rows.append(
                (
                    str(coordinate),
                    len(set(values)),
                    _fmt(min(values), 6),
                    _fmt(max(values), 6),
                )
            )
    protocol = manifest.get("training_protocol", {})
    metrics_protocol = manifest.get("metrics_protocol", {})
    datasets = manifest.get("datasets", {})
    split_key = "train_bars" if split_name in {"training", "train"} else "test_bars"
    evaluation_bars = datasets.get(split_key, {}) if isinstance(datasets, dict) else {}
    return {
        "coordinate_count": len(coordinates),
        "action_rows": len(action_rows),
        "exit_unique": len(set(exit_values)),
        "exit_min": min(exit_values) if exit_values else None,
        "exit_max": max(exit_values) if exit_values else None,
        "decoded_modes": decoded,
        "logged_modes": logged_modes,
        "mismatches": mismatches,
        "coordinate_rows": coordinate_rows,
        "all_action_coordinates_constant": bool(coordinate_rows)
        and all(row[1] == 1 for row in coordinate_rows),
        "distribution": protocol.get("action_distribution", "unknown"),
        "historical_test_role": metrics_protocol.get("historical_test_role", "unspecified"),
        "evaluation_bars_start": evaluation_bars.get("start", "unknown"),
        "evaluation_bars_end": evaluation_bars.get("end", "unknown"),
    }


def _run_analysis(run: dict[str, Any], near_be_r: float) -> dict[str, Any]:
    positions = run["positions"]
    events = run["events"]
    finals, terminal_by_position = _terminal_records(positions, events, near_be_r)
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    final_by_id = {record["position_id"]: record for record in finals}
    for record in finals:
        by_reason[record["exit_reason"]].append(record)

    tp_events = [event for event in events if event.get("type") in TP_EVENT_TYPES]
    tp_by_position: dict[str, set[str]] = defaultdict(set)
    for event in tp_events:
        if event.get("position_id"):
            tp_by_position[event["position_id"]].add(event.get("reason", ""))
    tp_levels: dict[str, dict[str, Any]] = {}
    for level in sorted({event.get("reason", "") for event in tp_events}):
        ids = {
            event.get("position_id")
            for event in tp_events
            if event.get("reason") == level and event.get("position_id")
        }
        level_records = [
            final_by_id[position_id] for position_id in ids if position_id in final_by_id
        ]
        tp_levels[level] = {
            "positions": len(ids),
            "final_reasons": Counter(record["exit_reason"] for record in level_records),
            "positive": sum(record["pnl"] > 0.0 for record in level_records),
            "negative": sum(record["pnl"] < 0.0 for record in level_records),
        }
    tp_combinations = Counter(
        "+".join(sorted(levels))
        for position_id, levels in tp_by_position.items()
        if position_id in final_by_id
    )

    swing_update_indices: dict[str, list[int]] = defaultdict(list)
    swing_updates = []
    tp_swing_updates = []
    for index, event in enumerate(events):
        if event.get("type") == "sl_update" and event.get("reason") == "confirmed_swing":
            swing_updates.append(event)
            if event.get("position_id"):
                swing_update_indices[event["position_id"]].append(index)
        if (
            event.get("type") == "tp_update"
            and event.get("reason") == "new_confirmed_favorable_swing"
        ):
            tp_swing_updates.append(event)
    swing_updated_positions = []
    for record in finals:
        position_id = record["position_id"]
        final_index = record["terminal_event_index"]
        if final_index is not None and any(
            update_index < final_index for update_index in swing_update_indices.get(position_id, [])
        ):
            swing_updated_positions.append(record)
    swing_updated_stop_closes = [
        record for record in swing_updated_positions if record["exit_reason"] == "structure_sl"
    ]

    action = _action_summary(run, run.get("split_name", "testing"))
    split = run["manifest"].get("split", {})
    learning = run["manifest"].get("learning_result", {})
    checkpoint = run["manifest"].get("checkpoint", {})
    config = run["manifest"].get("config", {})
    run["analysis"] = {
        "finals": finals,
        "by_reason": by_reason,
        "tp_events": tp_events,
        "tp_levels": tp_levels,
        "tp_combinations": tp_combinations,
        "tp_by_position": tp_by_position,
        "swing_updates": swing_updates,
        "swing_update_positions": swing_update_indices,
        "swing_updated_positions": swing_updated_positions,
        "swing_updated_stop_closes": swing_updated_stop_closes,
        "tp_swing_updates": tp_swing_updates,
        "action": action,
        "split": split,
        "split_name": run.get("split_name", "testing"),
        "seed": run["manifest"].get("seed", learning.get("seed", "unknown")),
        "dataset_id": learning.get("dataset_id", "unknown"),
        "checkpoint_sha": checkpoint.get("sha256", "unknown"),
        "config_sha": config.get("sha256", "unknown"),
        "manifest_sha": _sha256(run["manifest_path"]),
        "positions_sha": _sha256(run["positions_path"]),
        "trades_sha": _sha256(run["trades_path"]),
    }
    return run


def _render_run(run: dict[str, Any], near_be_r: float) -> list[str]:
    analysis = run["analysis"]
    action = analysis["action"]
    split_name = analysis["split_name"]
    split_info = analysis["split"]
    is_training = split_name in {"training", "train"}
    start_key = "train_start" if is_training else "test_start"
    end_key = "train_end" if is_training else "test_end"
    split_start = split_info.get(start_key, "unknown")
    split_end = split_info.get(end_key) or action["evaluation_bars_end"]
    role_text = " Test role `{}`.".format(action["historical_test_role"]) if not is_training else ""
    lines = ["### {}".format(_escape(run["label"]))]
    lines.append("")
    lines.append(
        "Seed `{}`; training action distribution `{}`; dataset `{}`. "
        "{} split spans `{}` to `{}`; saved bars span `{}` to `{}`.{}".format(
            analysis["seed"],
            action["distribution"],
            analysis["dataset_id"],
            split_name,
            split_start,
            split_end,
            action["evaluation_bars_start"],
            action["evaluation_bars_end"],
            role_text,
        )
    )
    lines.append("")
    lines.append(
        "Exit mapping in the environment: `exit_u >= {:.2f}` selects `ema_21`; "
        "otherwise it selects `structural`. Logged open rows: `{}`. "
        "Logged action-bearing rows (open/reject): `{}`; distinct `exit_u`: `{}`; "
        "range: `{}`–`{}`. Open-row decoded/logged mismatches: `{}`.".format(
            EMA_EXIT_THRESHOLD,
            dict(sorted(action["logged_modes"].items())),
            action["action_rows"],
            action["exit_unique"],
            _fmt(action["exit_min"], 6),
            _fmt(action["exit_max"], 6),
            action["mismatches"],
        )
    )
    if action["all_action_coordinates_constant"]:
        lines.append(
            "All {} logged action coordinates have one distinct value across the saved "
            "action-bearing rows. This indicates no observed action variation in this replay; "
            "it does not by itself rule out a logging or replay defect.".format(
                action["coordinate_count"]
            )
        )
    lines.append("")
    lines.append("| Action coordinate | Distinct normalized values | Min | Max |")
    lines.append("|---|---:|---:|---:|")
    for coordinate, unique, minimum, maximum in action["coordinate_rows"]:
        lines.append(f"| {coordinate} | {unique} | {minimum} | {maximum} |")
    lines.append("")

    lines.append(
        "| Terminal close reason | Positions | Net winners | Net losers | Near BE* | Win rate | Mean net PnL | Mean R |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    all_finals = analysis["finals"]
    for reason in sorted(analysis["by_reason"]):
        records = analysis["by_reason"][reason]
        positive = sum(record["pnl"] > 0.0 for record in records)
        negative = sum(record["pnl"] < 0.0 for record in records)
        near_be = sum(record["near_be"] for record in records)
        mean_pnl = statistics.mean(record["pnl"] for record in records)
        r_values = [record["realized_r"] for record in records if record["realized_r"] is not None]
        mean_r = statistics.mean(r_values) if r_values else None
        win_rate = positive / len(records) if records else 0.0
        lines.append(
            f"| {_escape(reason)} | {len(records)} | {positive} | {negative} | {near_be} | {win_rate:.1%} | {_fmt(mean_pnl, 2)} | {_fmt(mean_r, 3)} |"
        )
    lines.append(
        "| **Total** | **{}** | **{}** | **{}** | **{}** | **{:.1%}** | **{}** | **{}** |".format(
            len(all_finals),
            sum(record["pnl"] > 0.0 for record in all_finals),
            sum(record["pnl"] < 0.0 for record in all_finals),
            sum(record["near_be"] for record in all_finals),
            (
                sum(record["pnl"] > 0.0 for record in all_finals) / len(all_finals)
                if all_finals
                else 0.0
            ),
            _fmt(sum(record["pnl"] for record in all_finals), 2),
            _fmt(
                statistics.mean(
                    [
                        record["realized_r"]
                        for record in all_finals
                        if record["realized_r"] is not None
                    ]
                )
                if any(record["realized_r"] is not None for record in all_finals)
                else None,
                3,
            ),
        )
    )
    lines.append("")
    lines.append(
        f"*Near BE is a subset of the win/loss columns: `abs(final realized R) <= {near_be_r:.3f}`. "
        "It does not prove that the simulator's breakeven stop trigger fired."
    )
    lines.append("")

    lines.append("#### Take-profit events")
    lines.append("")
    if analysis["tp_events"]:
        lines.append("| TP event | Events | Unique positions | Final terminal outcomes |")
        lines.append("|---|---:|---:|---|")
        event_counts = Counter(
            (event.get("reason", "unknown"), event.get("type", "unknown"))
            for event in analysis["tp_events"]
        )
        for (reason, event_type), count in sorted(event_counts.items()):
            level = analysis["tp_levels"].get(reason, {})
            outcomes = Counter(
                record["exit_reason"]
                for record in analysis["finals"]
                if record["position_id"]
                in {
                    event.get("position_id")
                    for event in analysis["tp_events"]
                    if event.get("reason") == reason
                }
            )
            lines.append(
                "| {} / {} | {} | {} | {} |".format(
                    _escape(reason),
                    _escape(event_type),
                    count,
                    level.get("positions", 0),
                    _escape(dict(sorted(outcomes.items()))),
                )
            )
        lines.append("")
        lines.append(
            "TP event counts can overlap by position. Position-level TP combinations: `{}`.".format(
                dict(sorted(analysis["tp_combinations"].items()))
            )
        )
        lines.append("")
        for reason in sorted(analysis["tp_levels"]):
            ids = {
                event.get("position_id")
                for event in analysis["tp_events"]
                if event.get("reason") == reason
            }
            references: Counter[str] = Counter()
            for event in analysis["tp_events"]:
                if event.get("reason") == reason:
                    for family in _reference_families(event.get("tp_ref", "")):
                        references[family] += 1
            lines.append(
                "`{}` unique positions: {}; final terminal outcome: {}; target reference families "
                "(overlapping when composite): {}.".format(
                    _escape(reason),
                    len(ids),
                    _escape(dict(sorted(analysis["tp_levels"][reason]["final_reasons"].items()))),
                    _escape(dict(sorted(references.items()))),
                )
            )
    else:
        lines.append("No TP partial or full-close events were recorded.")
    lines.append("")

    lines.append("#### Swing stop/target management")
    lines.append("")
    update_positions = set(analysis["swing_update_positions"])
    swing_stop_closes = analysis["swing_updated_stop_closes"]
    lines.append(
        "Confirmed-swing SL updates: {} events across {} positions. Of those, {} later "
        "closed by `structure_sl`: {} net winners, {} net losers, {} near BE. "
        "TP updates after a new confirmed favorable swing: {} events.".format(
            len(analysis["swing_updates"]),
            len(update_positions),
            len(swing_stop_closes),
            sum(record["pnl"] > 0.0 for record in swing_stop_closes),
            sum(record["pnl"] < 0.0 for record in swing_stop_closes),
            sum(record["near_be"] for record in swing_stop_closes),
            len(analysis["tp_swing_updates"]),
        )
    )
    lines.append("")

    lines.append("#### Provenance")
    lines.append("")
    lines.append("| Input | SHA-256 |")
    lines.append("|---|---|")
    lines.append("| run_manifest.json | `{}` |".format(analysis["manifest_sha"]))
    lines.append("| positions.csv | `{}` |".format(analysis["positions_sha"]))
    lines.append("| trades.csv | `{}` |".format(analysis["trades_sha"]))
    lines.append("| config | `{}` |".format(analysis["config_sha"]))
    lines.append("| checkpoint | `{}` |".format(analysis["checkpoint_sha"]))
    lines.append("")
    return lines


def render_report(
    campaign_root: Path,
    runs: Sequence[dict[str, Any]],
    split: str,
    near_be_r: float,
) -> str:
    analyzer_path = Path(__file__).resolve()
    command = f"python3 {analyzer_path} --campaign-root {campaign_root} --split {split} --near-be-r-threshold {near_be_r}"
    lines = [
        "# Exit and Action Outcome Analysis",
        "",
        f"Campaign: `{campaign_root.name}`  ",
        f"Evaluation split: `{split}`  ",
        f"EMA exit threshold: `exit_u >= {EMA_EXIT_THRESHOLD:.2f}`  ",
        f"Near-BE outcome threshold: `abs(final realized R) <= {near_be_r:.3f}`",
        f"Analyzer SHA-256: `{_sha256(analyzer_path)}`",
        f"Reproduce: `{command}`",
        "",
        "This is a descriptive replay of saved evaluation artifacts. It does not load a model, "
        "re-evaluate a checkpoint, or establish generalization. TP1 partials are reported as "
        "events and may belong to positions whose later terminal exit was a stop, TP3, or session close.",
        "",
    ]
    for run in runs:
        lines.extend(_render_run(run, near_be_r))
    lines.extend(
        [
            "### Interpretation limits",
            "",
            "A single evaluation split and seed provide descriptive counts, not evidence that the "
            "exit policy generalizes. The binary EMA/structural threshold is a decoder; it does not "
            "make a Gaussian policy a calibrated categorical choice. Near-BE results are inferred "
            "from final realized R because the saved trade event stream does not label an exact "
            "breakeven-stop trigger as a terminal reason.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--split", default="testing")
    parser.add_argument("--near-be-r-threshold", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    campaign_root = args.campaign_root.expanduser().resolve()
    if not campaign_root.is_dir():
        raise FileNotFoundError(f"campaign root does not exist: {campaign_root}")
    if args.near_be_r_threshold < 0.0:
        raise ValueError("--near-be-r-threshold must be non-negative")
    runs = _discover_runs(campaign_root, args.split)
    for run in runs:
        run["split_name"] = args.split
        _run_analysis(run, args.near_be_r_threshold)
    output = args.out or (campaign_root / "reports" / f"exit_outcomes_{args.split}.md")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_report(campaign_root, runs, args.split, args.near_be_r_threshold), encoding="utf-8"
    )
    print(f"Analyzed {len(runs)} runs from {campaign_root} ({args.split})")
    print(f"Wrote {output}")
    for run in runs:
        action = run["analysis"]["action"]
        print(
            "{} seed={} trades={} exit_u_unique={} modes={}".format(
                run["label"],
                run["analysis"]["seed"],
                len(run["analysis"]["finals"]),
                action["exit_unique"],
                dict(sorted(action["logged_modes"].items())),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
