"""Manage immutable weekly multi-batch experiment artifacts and reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STRATEGIES = ("baseline", "po3_ifvg", "distribution")
Runner = Callable[["RunSpec"], dict[str, Any]]


@dataclass(frozen=True)
class BatchSpec:
    """Configuration shared by every run in one batch."""

    objective: str = "Weekly comparison"
    strategies: tuple[str, ...] = STRATEGIES
    seeds: tuple[int, ...] = (42,)
    steps: int = 50_000
    config: str | None = None
    dataset_hash: str = "unspecified"
    evaluation_protocol: str = "validation_only"
    train_end: str | None = None
    test_start: str | None = None


@dataclass(frozen=True)
class RunSpec:
    """Resolved run identity and comparability metadata."""

    week: int
    batch: int
    run_id: str
    strategy: str
    seed: int
    steps: int
    run_dir: Path
    comparability: dict[str, Any]
    retry_of: str | None = None
    config: str | None = None


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _relative_link(path: Path, report: Path) -> str:
    return (
        Path(path).relative_to(report.parent).as_posix()
        if path.is_relative_to(report.parent)
        else path.as_posix()
    )


def render_batch_report(batch_dir: Path, manifest: dict[str, Any]) -> str:
    """Render a Markdown report retaining successful and failed runs."""
    lines = [f"# Batch {manifest['batch']}", "", f"**Objective:** {manifest['objective']}", ""]
    lines.append("| Strategy | Seed | Status | Retry of | Artifacts |")
    lines.append("|---|---:|---|---|---|")
    for run in manifest.get("runs", []):
        link = _relative_link(batch_dir / run["run_id"], batch_dir / "report.md")
        lines.append(
            f"| {run['strategy']} | {run['seed']} | {run['status']} | "
            f"{run.get('retry_of') or '—'} | [{run['run_id']}]({link}) |"
        )
    lines.extend(["", "## Comparability", "", f"`{manifest['comparability_key']}`", ""])
    if manifest.get("incomparable_reasons"):
        lines.extend(["**Not comparable:** " + "; ".join(manifest["incomparable_reasons"]), ""])
    return "\n".join(lines)


def render_weekly_report(week_dir: Path, batches: Iterable[dict[str, Any]]) -> str:
    """Render a weekly index linking every batch and run."""
    batch_list = list(batches)
    lines = [
        f"# Week {week_dir.name.removeprefix('week_')}",
        "",
        "| Batch | Objective | Status | Report |",
        "|---:|---|---|---|",
    ]
    for batch in batch_list:
        batch_dir = week_dir / "batches" / f"batch_{batch['batch']}"
        link = _relative_link(batch_dir / "report.md", week_dir / "report.md")
        statuses = [run["status"] for run in batch.get("runs", [])]
        status = "failed" if any(s == "failed" for s in statuses) else "completed"
        if any(s == "running" for s in statuses):
            status = "incomplete"
        lines.append(f"| {batch['batch']} | {batch['objective']} | {status} | [report]({link}) |")
    lines.extend(["", "## Run artifacts", ""])
    for batch in batch_list:
        for run in batch.get("runs", []):
            path = week_dir / "batches" / f"batch_{batch['batch']}" / run["run_id"]
            link = _relative_link(path, week_dir / "report.md")
            lines.append(
                f"- `{run['strategy']}` seed {run['seed']} ({run['status']}): [{run['run_id']}]({link})"
            )
    return "\n".join(lines) + "\n"


@dataclass
class WeeklyOrchestrator:
    """Execute a matrix while preserving evidence for failures and retries."""

    output_root: Path = Path("outputs/thesis_progress")
    runner: Runner | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _batches: list[dict[str, Any]] = field(default_factory=list, init=False)

    def run_week(self, week: int, batches: Iterable[BatchSpec], *, retries: int = 0) -> Path:
        """Execute all batches and write weekly and batch manifests."""
        week_dir = self.output_root / f"week_{week}"
        (week_dir / "batches").mkdir(parents=True, exist_ok=True)
        self._batches = []
        for batch_number, spec in enumerate(batches, start=1):
            self._run_batch(week, batch_number, spec, week_dir, retries)
        _json_dump(week_dir / "manifest.json", {"week": week, "batches": self._batches})
        (week_dir / "report.md").write_text(render_weekly_report(week_dir, self._batches))
        return week_dir

    def _run_batch(
        self, week: int, number: int, spec: BatchSpec, week_dir: Path, retries: int
    ) -> None:
        if not set(spec.strategies).issubset(STRATEGIES):
            raise ValueError(f"unknown strategy: {sorted(set(spec.strategies) - set(STRATEGIES))}")
        batch_dir = week_dir / "batches" / f"batch_{number}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        comparable = asdict(spec)
        comparable.pop("objective", None)
        comparable["git_commit"] = _git_commit()
        manifest: dict[str, Any] = {
            "week": week,
            "batch": number,
            "objective": spec.objective,
            "comparability": comparable,
            "comparability_key": _hash(comparable),
            "runs": [],
        }
        _json_dump(batch_dir / "manifest.json", manifest)
        for strategy in spec.strategies:
            for seed in spec.seeds:
                previous: str | None = None
                for attempt in range(retries + 1):
                    run_id = self._run_id(strategy, seed, attempt, batch_dir)
                    run_dir = batch_dir / run_id
                    run_dir.mkdir()
                    run = RunSpec(
                        week,
                        number,
                        run_id,
                        strategy,
                        seed,
                        spec.steps,
                        run_dir,
                        comparable,
                        previous,
                        spec.config,
                    )
                    record = {**asdict(run), "run_dir": str(run_dir), "status": "running"}
                    record["comparability_key"] = manifest["comparability_key"]
                    _json_dump(run_dir / "metadata.json", record)
                    try:
                        result = (self.runner or self._default_runner)(run)
                        record.update(result)
                        record["status"] = result.get("status", "completed")
                    except Exception as exc:  # preserve one failure and continue matrix
                        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                    _json_dump(run_dir / "result.json", record)
                    manifest["runs"].append(record)
                    previous = run_id
                    if record["status"] != "failed":
                        break
        manifest["incomparable_reasons"] = self._incomparable_reasons(manifest["runs"])
        _json_dump(batch_dir / "manifest.json", manifest)
        (batch_dir / "report.md").write_text(render_batch_report(batch_dir, manifest))
        self._batches.append(manifest)

    def _run_id(self, strategy: str, seed: int, attempt: int, batch_dir: Path) -> str:
        stamp = self.clock().strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_r{attempt}" if attempt else ""
        base = f"run_{stamp}_{strategy}_s{seed}{suffix}"
        candidate = base
        index = 1
        while (batch_dir / candidate).exists():
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    @staticmethod
    def _incomparable_reasons(runs: list[dict[str, Any]]) -> list[str]:
        keys = {run.get("comparability_key") for run in runs if run.get("status") != "failed"}
        return ["comparability metadata differs"] if len(keys) > 1 else []

    @staticmethod
    def _default_runner(run: RunSpec) -> dict[str, Any]:
        """Run the canonical training entrypoint in the isolated run directory."""
        command = [
            "python",
            "-m",
            "quant_rl.train.train_rl",
            "--seed",
            str(run.seed),
            "--out",
            str(run.run_dir),
            "--strategy",
            run.strategy,
            f"ppo.total_timesteps={run.steps}",
        ]
        if run.config:
            command[5:5] = ["--config", run.config]
        (run.run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        (run.run_dir / "stdout.log").write_text(completed.stdout)
        (run.run_dir / "stderr.log").write_text(completed.stderr)
        if completed.returncode:
            return {"status": "failed", "returncode": completed.returncode}
        return {"status": "completed", "returncode": 0}
