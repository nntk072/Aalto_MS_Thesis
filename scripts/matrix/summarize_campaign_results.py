#!/usr/bin/env python3
"""Create a checksummed, non-ranking report from a campaign's saved run metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _run_dir(attempt_dir: Path, run_key: str) -> Path | None:
    candidates = sorted(
        (path for path in (attempt_dir / run_key).glob("*") if path.is_dir()),
        key=lambda path: path.name,
    )
    return candidates[-1] if candidates else None


def _account_currency(run_dir: Path | None) -> str | None:
    currency, _ = _account_config(run_dir)
    return currency


def _account_config(run_dir: Path | None) -> tuple[str | None, float | None]:
    if run_dir is None:
        return None, None
    try:
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None, None
    account = config.get("account") if isinstance(config, dict) else None
    if not isinstance(account, dict):
        return None, None
    currency = account.get("currency") if isinstance(account, dict) else None
    initial = account.get("initial_balance")
    try:
        initial_balance = float(initial) if initial is not None else None
    except (TypeError, ValueError):
        initial_balance = None
    return (str(currency) if currency is not None else None), initial_balance


def _pnl_reconciliation(
    run_dir: Path | None,
    split: str,
    metrics: dict[str, Any] | None,
    initial_balance: float | None,
) -> dict[str, Any]:
    """Check saved split PnL against its final equity value."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "initial_balance": initial_balance,
        "final_equity": None,
        "equity_delta": None,
        "reported_total_pnl": None,
        "difference": None,
        "tolerance": 0.02,
    }
    if run_dir is None or metrics is None or initial_balance is None:
        return result
    try:
        with (run_dir / split / "equity.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return result
        final_equity = float(rows[-1]["equity"])
        reported_pnl = float(metrics["total_pnl"])
    except (OSError, csv.Error, KeyError, TypeError, ValueError, IndexError):
        return result
    if not math.isfinite(final_equity) or not math.isfinite(reported_pnl):
        return result
    equity_delta = final_equity - initial_balance
    difference = equity_delta - reported_pnl
    result.update(
        final_equity=final_equity,
        equity_delta=equity_delta,
        reported_total_pnl=reported_pnl,
        difference=difference,
        status="verified" if abs(difference) <= result["tolerance"] else "mismatch",
    )
    return result


def _resource_profiles(campaign_dir: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted((campaign_dir / "logs").glob("resources_*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                header = set(reader.fieldnames or ())
        except OSError:
            continue
        active = [row for row in rows if row.get("campaign_active") == "1"]
        pss_values = []
        for row in active:
            try:
                pss_values.append(int(row["process_tree_pss_bytes"]))
            except (KeyError, ValueError, TypeError):
                continue
        gpu_util: list[list[int]] = [[], []]
        gpu_mem: list[list[int]] = [[], []]
        for row in active:
            parsed: list[tuple[int, int]] = []
            if "gpu_utilization_pct_by_device" in header:
                try:
                    utils = json.loads(row.get("gpu_utilization_pct_by_device", "[]"))
                    memories = json.loads(row.get("gpu_memory_used_mib_by_device", "[]"))
                    parsed = [(int(u), int(m)) for u, m in zip(utils, memories)]
                except (ValueError, TypeError):
                    parsed = []
            elif "gpu_utilization_pct" in header and "gpu_memory_used_mib" in header:
                combined = (
                    row.get("gpu_utilization_pct", "") + "," + row.get("gpu_memory_used_mib", "")
                )
                for line in combined.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2:
                        try:
                            parsed.append((int(fields[0]), int(fields[1])))
                        except ValueError:
                            continue
            if len(parsed) == 2:
                for index, (util, memory) in enumerate(parsed):
                    gpu_util[index].append(util)
                    gpu_mem[index].append(memory)
        profiles.append(
            {
                "resource_csv": str(path.relative_to(campaign_dir)),
                "sha256": _sha256(path),
                "sample_count": len(rows),
                "active_sample_count": len(active),
                "first_timestamp_utc": rows[0].get("timestamp_utc") if rows else None,
                "last_timestamp_utc": rows[-1].get("timestamp_utc") if rows else None,
                "last_campaign_active": rows[-1].get("campaign_active") if rows else None,
                "active_process_tree_pss_mean_gib": (
                    sum(pss_values) / len(pss_values) / (1024**3) if pss_values else None
                ),
                "active_process_tree_pss_peak_gib": (
                    max(pss_values) / (1024**3) if pss_values else None
                ),
                "gpu_samples_are_reconstructed_from_raw_rows": (
                    "gpu_utilization_pct_by_device" not in header
                ),
                "gpu_devices": [
                    {
                        "device_index": index,
                        "mean_utilization_pct": (
                            sum(gpu_util[index]) / len(gpu_util[index]) if gpu_util[index] else None
                        ),
                        "peak_utilization_pct": max(gpu_util[index]) if gpu_util[index] else None,
                        "mean_memory_used_mib": (
                            sum(gpu_mem[index]) / len(gpu_mem[index]) if gpu_mem[index] else None
                        ),
                        "peak_memory_used_mib": max(gpu_mem[index]) if gpu_mem[index] else None,
                        "sample_count": len(gpu_util[index]),
                    }
                    for index in range(2)
                ],
                "process_tree_cpu_time_sampled": "process_tree_cpu_time_usec" in header,
                "allocation_cgroup_scope_sampled": "cgroup_memory_scope" in header,
            }
        )
    return profiles


def _split_log(log_path: Path, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"survived_full_period": None, "days_traded": None, "days_total": None}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result
    pattern = re.compile(
        rf"\[{re.escape(label)}(?:/[^\]]+)?\].*?survived=(True|False)\s+days=(\d+)/(\d+)"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            result = {
                "survived_full_period": match.group(1) == "True",
                "days_traded": int(match.group(2)),
                "days_total": int(match.group(3)),
            }
            break
    return result


def _finite_metrics(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, float) and not math.isfinite(item):
            cleaned[key] = None
        else:
            cleaned[key] = item
    return cleaned


def build_summary(campaign_dir: Path) -> dict[str, Any]:
    campaign = _read_json(campaign_dir / "campaign.json")
    status = _read_json(campaign_dir / "reports" / "campaign_status.json")
    if campaign is None or status is None:
        raise ValueError("campaign.json and reports/campaign_status.json are required")

    jobs: list[dict[str, Any]] = []
    for job in campaign.get("jobs", []):
        if not isinstance(job, dict):
            continue
        run_key = str(job.get("key", ""))
        if not run_key:
            continue
        attempt_number = int(job.get("attempts", 1))
        attempt_dir = campaign_dir / "runs" / "attempts" / run_key / f"attempt{attempt_number}"
        run_dir = _run_dir(attempt_dir, run_key)
        train_path = run_dir / "training" / "metrics.json" if run_dir else Path()
        test_path = run_dir / "testing" / "metrics.json" if run_dir else Path()
        gate_path = run_dir / "train_equity_gate.json" if run_dir else Path()
        train_metrics = _read_json(train_path) if run_dir else None
        test_metrics = _read_json(test_path) if run_dir else None
        gate = _read_json(gate_path) if run_dir else None
        account_currency, initial_balance = _account_config(run_dir)
        log_path = campaign_dir / str(job.get("log", ""))
        if not log_path.is_file():
            log_path = campaign_dir / "logs" / "workers" / f"{run_key}.log"
        job_report = {
            "run_key": run_key,
            "variant": job.get("variant"),
            "seed": job.get("seed"),
            "controller_status": job.get("status"),
            "exit_code": job.get("exit_code"),
            "failure_reason": job.get("failure_reason"),
            "attempt": attempt_number,
            "account_currency": account_currency,
            "train_pnl_reconciliation": _pnl_reconciliation(
                run_dir, "training", train_metrics, initial_balance
            ),
            "test_pnl_reconciliation": _pnl_reconciliation(
                run_dir, "testing", test_metrics, initial_balance
            ),
            "train_equity_gate": gate,
            "train_metrics": _finite_metrics(train_metrics),
            "test_metrics_report_only": _finite_metrics(test_metrics),
            "train_replay": _split_log(log_path, "train"),
            "test_replay_report_only": _split_log(log_path, "test"),
            "evidence_sha256": {
                "attempt_provenance.json": _sha256(attempt_dir / "attempt_provenance.json"),
                "training/metrics.json": _sha256(train_path) if run_dir else None,
                "training/equity.csv": _sha256(run_dir / "training" / "equity.csv")
                if run_dir
                else None,
                "testing/metrics.json": _sha256(test_path) if run_dir else None,
                "testing/equity.csv": _sha256(run_dir / "testing" / "equity.csv")
                if run_dir
                else None,
                "train_equity_gate.json": _sha256(gate_path) if run_dir else None,
                "worker_log": _sha256(log_path),
            },
            "run_artifact_dir": str(run_dir.relative_to(campaign_dir)) if run_dir else None,
            "worker_log": str(log_path.relative_to(campaign_dir))
            if log_path.is_relative_to(campaign_dir)
            else str(log_path),
        }
        jobs.append(job_report)

    evaluation = campaign.get("evaluation", {})
    counts = status.get("statuses", {})
    train_gate_failures = sum(
        item.get("train_equity_gate", {}).get("ok") is False
        for item in jobs
        if isinstance(item.get("train_equity_gate"), dict)
    )
    train_gate_reason_counts: Counter[str] = Counter()
    for item in jobs:
        gate = item.get("train_equity_gate")
        if not isinstance(gate, dict) or gate.get("ok") is not False:
            continue
        reasons = [part.strip() for part in str(gate.get("reason") or "unspecified").split(",")]
        train_gate_reason_counts.update(reason for reason in reasons if reason)
    test_not_survived = sum(
        item.get("test_replay_report_only", {}).get("survived_full_period") is False
        for item in jobs
    )
    test_low_activity = sum(
        item.get("test_replay_report_only", {}).get("survived_full_period") is True
        and isinstance(item.get("test_metrics_report_only"), dict)
        and int(item["test_metrics_report_only"].get("n_trades", 0)) <= 2
        for item in jobs
    )
    pnl_reconciliation_counts = {
        "verified": 0,
        "mismatch": 0,
        "unavailable": 0,
    }
    for item in jobs:
        for key in ("train_pnl_reconciliation", "test_pnl_reconciliation"):
            status_value = item[key]["status"]
            pnl_reconciliation_counts[status_value] += 1
    return {
        "schema_version": 1,
        "campaign_dir": str(campaign_dir),
        "campaign_created_utc": campaign.get("created"),
        "campaign_job_id": campaign.get("job_id"),
        "campaign_git_sha": campaign.get("git_sha"),
        "campaign_fingerprint": campaign.get("fingerprint"),
        "requested_steps_per_variant": campaign.get("timesteps"),
        "seed": 42 if jobs and all(item.get("seed") == 42 for item in jobs) else None,
        "data_source": campaign.get("data_source"),
        "n_envs": campaign.get("n_envs"),
        "worker_cpus": campaign.get("worker_cpus"),
        "worker_gpus_per_node": campaign.get("worker_gpus_per_node"),
        "worker_nodes": campaign.get("worker_nodes"),
        "evaluation_protocol": evaluation,
        "controller_status_as_of_utc": status.get("timestamp"),
        "controller_status_counts": counts,
        "resource_profiles": _resource_profiles(campaign_dir),
        "gate_failure_count": train_gate_failures,
        "train_gate_failure_reason_counts": dict(sorted(train_gate_reason_counts.items())),
        "test_non_survival_count": test_not_survived,
        "test_survived_low_activity_count": test_low_activity,
        "pnl_reconciliation_counts": pnl_reconciliation_counts,
        "campaign_manifest_sha256": _sha256(campaign_dir / "campaign.json"),
        "controller_status_sha256": _sha256(campaign_dir / "reports" / "campaign_status.json"),
        "summary_script_sha256": _sha256(Path(__file__).resolve()),
        "jobs": jobs,
    }


def _fmt_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:+.2f}%"


def _fmt_money(value: Any) -> str:
    return f"{value:+,.2f}" if isinstance(value, (int, float)) else "—"


def render_markdown(summary: dict[str, Any]) -> str:
    jobs = summary["jobs"]
    currencies = {job.get("account_currency") for job in jobs if job.get("account_currency")}
    pnl_label = next(iter(currencies)) if len(currencies) == 1 else "account currency"
    lines = [
        "# Campaign results snapshot",
        "",
        f"**Controller status as of:** {summary.get('controller_status_as_of_utc')}",
        f"**Slurm job:** {summary.get('campaign_job_id')}  ",
        f"**Specification:** {summary.get('requested_steps_per_variant')} steps per variant; seed {summary.get('seed')}; "
        f"{summary.get('n_envs')} environments; {summary.get('data_source')} backend.",
        f"**Evaluation role:** {summary.get('evaluation_protocol', {}).get('locked_oos_role')} "
        f"({summary.get('evaluation_protocol', {}).get('locked_oos_exposure')}).",
        "",
        "This is a descriptive snapshot, not a ranking. Training values are in-sample. Test values are report-only on the previously inspected split and are not confirmatory evidence.",
        "",
        "## Outcomes",
        "",
        f"- Controller counts: `{json.dumps(summary.get('controller_status_counts', {}), sort_keys=True)}`",
        f"- Train-equity gate failures with saved gate records: **{summary.get('gate_failure_count')} / {len(jobs)}**",
        "- Gate failure reasons (overlapping): "
        + (
            ", ".join(
                f"`{reason}` {count}/{len(jobs)}"
                for reason, count in summary.get("train_gate_failure_reason_counts", {}).items()
            )
            or "none recorded"
        ),
        f"- Test replays that did not survive the full period: **{summary.get('test_non_survival_count')} / {len(jobs)}**",
        f"- Test replays that survived with at most two trades: **{summary.get('test_survived_low_activity_count')} / {len(jobs)}**",
        f"- Split PnL reconciled against final equity: **{summary.get('pnl_reconciliation_counts', {}).get('verified', 0)} verified**, **{summary.get('pnl_reconciliation_counts', {}).get('mismatch', 0)} mismatched**, **{summary.get('pnl_reconciliation_counts', {}).get('unavailable', 0)} unavailable**",
        "",
        "## Per-variant metrics",
        "",
        f"PnL is in the account currency ({pnl_label}). `Survived / days` comes from the matching worker log.",
        "",
        f"| Variant | Controller | Train gate | Train return | Train PnL ({pnl_label}) | Train trades | Test return | Test PnL ({pnl_label}) | Test max DD | Test Sharpe | Test PF | Test trades | Test survived / days | Evidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for job in jobs:
        train = job.get("train_metrics") or {}
        test = job.get("test_metrics_report_only") or {}
        gate = job.get("train_equity_gate") or {}
        replay = job.get("test_replay_report_only") or {}
        days = "—"
        if replay.get("days_traded") is not None and replay.get("days_total") is not None:
            days = f"{replay['days_traded']}/{replay['days_total']}"
        survived = replay.get("survived_full_period")
        survived_label = "—" if survived is None else ("yes" if survived else "no")
        evidence_links: list[str] = []
        worker_log = job.get("worker_log")
        if worker_log:
            evidence_links.append(f"[worker log](../{worker_log})")
        run_artifact_dir = job.get("run_artifact_dir")
        if run_artifact_dir:
            evidence_links.append(f"[run files](../{run_artifact_dir})")
            evaluation_summary = (
                Path(summary["campaign_dir"]) / run_artifact_dir / "evaluations" / "summary.md"
            )
            if evaluation_summary.is_file():
                evidence_links.append(
                    f"[train/test evaluation](../{run_artifact_dir}/evaluations/summary.md)"
                )
        evidence = " · ".join(evidence_links) or "—"
        lines.append(
            "| {variant} | {status} | {gate} | {tr} | {tpnl} | {tn} | {er} | {epnl} | {edd} | {esharpe} | {epf} | {en} | {survived} / {days} | {evidence} |".format(
                variant=job.get("variant") or job.get("run_key"),
                status=job.get("controller_status", "—"),
                gate="PASS"
                if gate.get("ok") is True
                else "FAIL"
                if gate.get("ok") is False
                else "—",
                tr=_fmt_percent(train.get("total_return_pct")),
                tpnl=_fmt_money(train.get("total_pnl")),
                tn=train.get("n_trades", "—"),
                er=_fmt_percent(test.get("total_return_pct")),
                epnl=_fmt_money(test.get("total_pnl")),
                edd=(
                    f"{100 * test['max_drawdown']:.2f}%"
                    if isinstance(test.get("max_drawdown"), (int, float))
                    else "—"
                ),
                esharpe=f"{test['sharpe']:.3f}"
                if isinstance(test.get("sharpe"), (int, float))
                else "—",
                epf=f"{test['profit_factor']:.3f}"
                if isinstance(test.get("profit_factor"), (int, float))
                else "—",
                en=test.get("n_trades", "—"),
                survived=survived_label,
                days=days,
                evidence=evidence,
            )
        )
    lines.extend(
        [
            "",
            "## Resource samples",
            "",
        ]
    )
    for profile in summary.get("resource_profiles", []):
        lines.extend(
            [
                f"- `{profile['resource_csv']}`: {profile['active_sample_count']} active of {profile['sample_count']} samples; process-tree PSS mean {profile['active_process_tree_pss_mean_gib']:.1f} GiB and peak {profile['active_process_tree_pss_peak_gib']:.1f} GiB.",
                "- Per-device GPU utilization and memory are reconstructed from the sampler's raw two-row `nvidia-smi` output; the CSV does not have separate per-device columns. This sampler did not record process-tree CPU time or allocation-wide cgroup memory scope.",
            ]
        )
        for device in profile["gpu_devices"]:
            if device["sample_count"]:
                lines.append(
                    f"- GPU {device['device_index']}: mean utilization {device['mean_utilization_pct']:.1f}%, peak {device['peak_utilization_pct']}%; memory mean {device['mean_memory_used_mib']:.0f} MiB, peak {device['peak_memory_used_mib']} MiB."
                )
    lines.extend(
        [
            "",
            "## Reproducibility and limits",
            "",
            f"- Campaign manifest SHA-256: `{summary.get('campaign_manifest_sha256')}`",
            f"- Controller status SHA-256: `{summary.get('controller_status_sha256')}`",
            f"- Summary script SHA-256: `{summary.get('summary_script_sha256')}`",
            "- Each row includes SHA-256 values for its attempt provenance, split metrics and equity curves, train gate, and worker log in the adjacent JSON file.",
            "- Launch-time source-hash comparison with the current repository is recorded in [the source-fidelity audit](source_fidelity_audit.md).",
            "- This is a 100k-step seed-42 resource probe, not the frozen 20M thesis campaign. It cannot establish strategy viability, statistical significance, or confirmatory OOS performance.",
            "- Do not select a variant from this report. Preserve all failed gates and low/no-trade outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--out-stem", type=Path, default=None)
    args = parser.parse_args()
    campaign_dir = args.campaign_dir.resolve()
    out_stem = args.out_stem or campaign_dir / "reports" / "campaign_results_summary"
    summary = build_summary(campaign_dir)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    out_stem.with_suffix(".md").write_text(render_markdown(summary), encoding="utf-8")
    print(out_stem.with_suffix(".md"))
    print(out_stem.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
