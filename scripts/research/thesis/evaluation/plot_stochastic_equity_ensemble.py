#!/usr/bin/env python3
"""Replay completed campaign checkpoints stochastically and plot equity ensembles.

Each replay uses the same explicit seed list across source variants. The seed
controls sampled policy actions and any seeded execution randomness exposed by
the existing reanalysis interface. The median line is a pointwise summary; it
is not an individual policy trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DEFAULT_SEEDS = tuple(range(1001, 1021))
SPLITS = ("training", "testing")
REQUIRED_SPLIT_FILES = ("equity.csv", "metrics.json")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not LABEL_PATTERN.fullmatch(label) or not raw_path:
        raise argparse.ArgumentTypeError("--run must use LABEL=/absolute/path/to/run")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"source run does not exist: {path}")
    return label, path


def _campaign_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    for parent in (run_dir, *run_dir.parents):
        candidate = parent / "campaign.json"
        if candidate.is_file():
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"campaign manifest must be an object: {candidate}")
            jobs = value.get("jobs")
            if not isinstance(jobs, list) or not jobs:
                raise ValueError(f"campaign manifest has no job records: {candidate}")
            unfinished = [
                str(job.get("key", "unknown")) if isinstance(job, dict) else str(job)
                for job in jobs
                if not isinstance(job, dict) or job.get("status") != "succeeded"
            ]
            if unfinished:
                raise ValueError(
                    f"source campaign is not fully successful; unfinished jobs: "
                    f"{', '.join(unfinished)}"
                )
            return candidate, value
    raise ValueError(f"could not find a completed campaign.json above {run_dir}")


def _source_identity(label: str, run_dir: Path) -> dict[str, Any]:
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"source run manifest is missing: {run_manifest_path}")
    campaign_path, campaign = _campaign_manifest(run_dir)
    from evaluation.reanalysis.checkpoint import resolve_checkpoint

    checkpoint = resolve_checkpoint(run_dir, None).resolve()
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"source run manifest must be an object: {run_manifest_path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"source run manifest lacks dataset identities: {run_manifest_path}")
    return {
        "label": label,
        "source_run": str(run_dir),
        "source_run_manifest": str(run_manifest_path),
        "source_run_manifest_sha256": _sha256(run_manifest_path),
        "source_datasets": datasets,
        "campaign_manifest": str(campaign_path.resolve()),
        "campaign_manifest_sha256": _sha256(campaign_path),
        "campaign_fingerprint": campaign.get("fingerprint"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }


def _read_replay(output: Path, source: dict[str, Any], seed: int) -> dict[str, Any]:
    manifest_path = output / "reanalysis_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(
            f"an incomplete replay output already exists; preserving it: {output}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "source_run": source["source_run"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "policy_action_mode": "stochastic",
        "seed": seed,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"replay identity mismatch in {manifest_path}: {', '.join(mismatches)}")

    outputs: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        split_outputs: dict[str, str] = {}
        for name in REQUIRED_SPLIT_FILES:
            path = output / split / name
            if not path.is_file():
                raise FileNotFoundError(f"required replay output is missing: {path}")
            split_outputs[name] = _sha256(path)
        outputs[split] = split_outputs
    return {
        "label": source["label"],
        "seed": seed,
        "output": str(output.resolve()),
        "reanalysis_manifest": str(manifest_path.resolve()),
        "reanalysis_manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "action_mode": manifest["policy_action_mode"],
        "action_seed": manifest["seed"],
        "execution_seed": manifest.get("execution_seed"),
        "dataset_identity": manifest.get("source_datasets"),
        "evaluated_datasets": manifest.get("evaluated_datasets"),
        "initial_balance": manifest.get("metrics_protocol", {}).get("initial_balance"),
        "outputs": outputs,
    }


def _load_equity(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "equity" not in frame.columns or frame.empty:
        raise ValueError(f"equity CSV is empty or lacks its equity column: {path}")
    index = pd.DatetimeIndex(frame.index)
    values = pd.to_numeric(frame["equity"], errors="coerce")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"equity timestamps must be unique and increasing: {path}")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"equity path contains non-finite values: {path}")
    return pd.Series(values.to_numpy(dtype=float), index=index, name=path.parent.name)


def _align_paths(paths: list[pd.Series], labels: list[str]) -> tuple[pd.DataFrame, list[int]]:
    if not paths:
        raise ValueError("at least one stochastic replay is required")
    longest_index = max(paths, key=len).index
    if any(path.index[0] != longest_index[0] for path in paths):
        raise ValueError("stochastic replay equity paths do not share a start timestamp")
    aligned: list[pd.Series] = []
    terminal_lengths: list[int] = []
    for path, label in zip(paths, labels, strict=True):
        if not path.index.equals(longest_index[: len(path)]):
            raise ValueError(
                f"{label}: equity timestamps are not an exact prefix of the shared dataset"
            )
        terminal_lengths.append(len(path))
        aligned.append(path.reindex(longest_index).ffill())
    return pd.concat(aligned, axis=1), terminal_lengths


def _plot_split(
    label: str,
    split: str,
    replay_records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    paths = [
        _load_equity(Path(record["output"]) / split / "equity.csv") for record in replay_records
    ]
    replay_labels = [f"seed {record['seed']}" for record in replay_records]
    matrix, terminal_lengths = _align_paths(paths, replay_labels)
    values = matrix.to_numpy(dtype=float)
    median = np.median(values, axis=1)
    lower = np.quantile(values, 0.25, axis=1)
    upper = np.quantile(values, 0.75, axis=1)
    initial_balance = float(replay_records[0]["initial_balance"])

    fig, ax = plt.subplots(figsize=(15, 6))
    for column in range(len(replay_records)):
        ax.plot(
            matrix.index,
            values[:, column],
            color="#667085",
            alpha=0.22,
            linewidth=0.75,
            label="Individual stochastic replays" if column == 0 else None,
        )
        terminal = terminal_lengths[column]
        if terminal < len(matrix.index):
            ax.scatter(
                matrix.index[terminal - 1],
                values[terminal - 1, column],
                marker="x",
                s=22,
                color="#667085",
                alpha=0.75,
                label="Replay ended early"
                if not any(length < len(matrix.index) for length in terminal_lengths[:column])
                else None,
                zorder=3,
            )
    ax.fill_between(
        matrix.index,
        lower,
        upper,
        color="#5271ff",
        alpha=0.2,
        label="25th–75th percentile",
        zorder=2,
    )
    ax.plot(
        matrix.index,
        median,
        color="#b42318",
        linewidth=2.4,
        label="Pointwise median stochastic equity",
        zorder=4,
    )
    ax.axhline(
        initial_balance,
        color="#98a2b3",
        linewidth=0.9,
        linestyle="--",
        label="Initial balance",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    ax.set_title(f"{label}: {split} equity across {len(paths)} stochastic replays")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    figure_path = output_dir / f"{label}_{split}_stochastic_equity.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    peaks = np.maximum.accumulate(values, axis=0)
    if np.any(peaks <= 0):
        raise ValueError(
            f"{label} {split}: cannot calculate percentage drawdown at non-positive equity"
        )
    drawdowns = (values / peaks - 1.0) * 100.0
    drawdown_median = np.median(drawdowns, axis=1)
    drawdown_lower = np.quantile(drawdowns, 0.25, axis=1)
    drawdown_upper = np.quantile(drawdowns, 0.75, axis=1)

    fig, ax = plt.subplots(figsize=(15, 5))
    for column in range(len(replay_records)):
        ax.plot(
            matrix.index,
            drawdowns[:, column],
            color="#667085",
            alpha=0.2,
            linewidth=0.7,
            label="Individual stochastic replays" if column == 0 else None,
        )
    ax.fill_between(
        matrix.index,
        drawdown_lower,
        drawdown_upper,
        color="#5271ff",
        alpha=0.2,
        label="25th–75th percentile",
    )
    ax.plot(
        matrix.index,
        drawdown_median,
        color="#b42318",
        linewidth=2.2,
        label="Pointwise median drawdown",
    )
    ax.axhline(0.0, color="#98a2b3", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # type: ignore[no-untyped-call]
    fig.autofmt_xdate()
    ax.set_title(f"{label}: {split} drawdown across {len(paths)} stochastic replays")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown from running peak")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    drawdown_figure = output_dir / f"{label}_{split}_stochastic_drawdown.png"
    fig.savefig(drawdown_figure, dpi=160)
    plt.close(fig)

    final_values = values[-1]
    returns_pct = (final_values / initial_balance - 1.0) * 100.0
    maximum_drawdowns_pct = np.min(drawdowns, axis=0)
    seed_positions = np.arange(len(replay_records))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    outcomes = (
        (axes[0], returns_pct, "Final return (%)", "Return"),
        (axes[1], -maximum_drawdowns_pct, "Maximum drawdown (%)", "Max drawdown"),
    )
    for axis, data, ylabel, title in outcomes:
        q25, med, q75 = np.quantile(data, [0.25, 0.5, 0.75])
        axis.axhspan(q25, q75, color="#5271ff", alpha=0.16, label="25th–75th percentile")
        axis.axhline(med, color="#b42318", linewidth=1.8, label="Median")
        axis.scatter(seed_positions, data, color="#344054", s=24, zorder=3)
        if title == "Return":
            axis.axhline(0.0, color="#98a2b3", linewidth=0.8, linestyle="--")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xticks(seed_positions)
    axes[-1].set_xticklabels([str(record["seed"]) for record in replay_records], rotation=45)
    axes[-1].set_xlabel("Stochastic replay seed")
    fig.suptitle(f"{label}: {split} outcome spread by replay")
    fig.tight_layout()
    outcomes_figure = output_dir / f"{label}_{split}_stochastic_outcomes.png"
    fig.savefig(outcomes_figure, dpi=160)
    plt.close(fig)

    summary = {
        "label": label,
        "split": split,
        "replay_count": len(paths),
        "seeds": [record["seed"] for record in replay_records],
        "start": str(matrix.index[0]),
        "end": str(matrix.index[-1]),
        "early_terminal_replays": [
            replay_records[index]["seed"]
            for index, length in enumerate(terminal_lengths)
            if length < len(matrix.index)
        ],
        "final_equity": {
            "median": float(np.median(final_values)),
            "p25": float(np.quantile(final_values, 0.25)),
            "p75": float(np.quantile(final_values, 0.75)),
            "profitable_replay_fraction": float(np.mean(final_values > initial_balance)),
        },
        "return_pct": {
            "median": float(np.median(returns_pct)),
            "p25": float(np.quantile(returns_pct, 0.25)),
            "p75": float(np.quantile(returns_pct, 0.75)),
        },
        "maximum_drawdown_pct": {
            "median": float(np.median(-maximum_drawdowns_pct)),
            "p25": float(np.quantile(-maximum_drawdowns_pct, 0.25)),
            "p75": float(np.quantile(-maximum_drawdowns_pct, 0.75)),
        },
        "figures": {
            "equity": {
                "path": str(figure_path.resolve()),
                "sha256": _sha256(figure_path),
            },
            "drawdown": {
                "path": str(drawdown_figure.resolve()),
                "sha256": _sha256(drawdown_figure),
            },
            "outcomes": {
                "path": str(outcomes_figure.resolve()),
                "sha256": _sha256(outcomes_figure),
            },
        },
    }
    _write_json(output_dir / f"{label}_{split}_stochastic_equity_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run seeded stochastic checkpoint replays and draw individual equity paths, "
            "their pointwise median, and the 25th–75th percentile band."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Completed campaign run; repeat once per variant.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New report directory, or an existing directory with this tool's manifest.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Fixed sampled-action replay seeds shared by every variant (default: 1001–1020).",
    )
    parser.add_argument(
        "--allow-historical",
        action="store_true",
        help="Acknowledge that the historical test split has already been inspected.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.allow_historical:
        raise SystemExit("pass --allow-historical to acknowledge the inspected test split")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds requires at least two unique integers")
    runs = [_parse_run(value) for value in args.run]
    labels = [label for label, _ in runs]
    if len(set(labels)) != len(labels):
        raise SystemExit("--run labels must be unique")

    sources = [_source_identity(label, path) for label, path in runs]
    dataset_identities = [
        json.dumps(source["source_datasets"], sort_keys=True) for source in sources
    ]
    if len(set(dataset_identities)) != 1:
        raise ValueError("source variants do not share the same recorded dataset identity")

    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        manifest_path = output_root / "ensemble_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"refusing to use an unrecognized output directory: {output_root}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sources") != sources or manifest.get("seeds") != args.seeds:
            raise FileExistsError(
                f"existing ensemble output has a different identity: {output_root}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "artifact_role": "stochastic_equity_ensemble",
            "sources": sources,
            "seeds": args.seeds,
            "policy_action_mode": "stochastic",
            "seed_controls": "sampled policy actions and seeded execution randomness",
            "median_definition": "pointwise median across replay equity values; not a replay",
            "band_definition": "pointwise 25th to 75th percentile across replay equity values",
            "historical_holdout_inspected": True,
            "replays": [],
            "summaries": [],
        }
        _write_json(output_root / "ensemble_manifest.json", manifest)

    replay_root = output_root / "replays"
    replay_root.mkdir(exist_ok=True)
    completed: dict[tuple[str, int], dict[str, Any]] = {
        (str(record["label"]), int(record["seed"])): record
        for record in manifest.get("replays", [])
    }
    for source in sources:
        for seed in args.seeds:
            key = (source["label"], seed)
            replay_output = replay_root / source["label"] / f"seed_{seed}"
            if not replay_output.exists():
                replay_output.parent.mkdir(exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "evaluation.reanalysis.runner",
                    "--run",
                    source["source_run"],
                    "--output",
                    str(replay_output),
                    "--checkpoint",
                    source["checkpoint"],
                    "--allow-historical",
                    "--stochastic",
                    "--seed",
                    str(seed),
                    "--max-charts",
                    "0",
                    "--no-save-html",
                ]
                print(f"Replay {source['label']} with stochastic seed {seed}", flush=True)
                subprocess.run(command, check=True)
            replay_record = _read_replay(replay_output, source, seed)
            if key in completed and completed[key] != replay_record:
                raise ValueError(f"replay ledger changed for {source['label']} seed {seed}")
            if key not in completed:
                manifest["replays"].append(replay_record)
                completed[key] = replay_record
                _write_json(output_root / "ensemble_manifest.json", manifest)
            if _sha256(Path(source["checkpoint"])) != source["checkpoint_sha256"]:
                raise ValueError(f"source checkpoint changed during replay: {source['checkpoint']}")

    all_replays = [completed[(label, seed)] for label in labels for seed in args.seeds]
    all_balances = {float(record["initial_balance"]) for record in all_replays}
    all_datasets = {
        json.dumps((record["dataset_identity"], record["evaluated_datasets"]), sort_keys=True)
        for record in all_replays
    }
    if len(all_balances) != 1 or len(all_datasets) != 1:
        raise ValueError("replays do not share the same balance and evaluated datasets")

    summaries: list[dict[str, Any]] = []
    figures_dir = output_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    for label in labels:
        label_replays = [completed[(label, seed)] for seed in args.seeds]
        for split in SPLITS:
            summaries.append(_plot_split(label, split, label_replays, figures_dir))

    manifest["summaries"] = summaries
    manifest["complete"] = True
    _write_json(output_root / "ensemble_manifest.json", manifest)
    print(f"Stochastic equity plots and provenance written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
