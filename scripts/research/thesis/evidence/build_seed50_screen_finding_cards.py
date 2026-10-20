#!/usr/bin/env python3
"""Package summary-derived finding cards for the unrecoverable seed-50 screen.

These cards preserve the campaign summary and dispositions. They are not
canonical numerical report packs and must not be used for variant selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configuration.paths import repository_root

ROOT = repository_root(__file__)
CAMPAIGN_DIR = (
    ROOT / "outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016"
)
SUMMARY_PATH = CAMPAIGN_DIR / "reports/campaign_results_summary_v2.json"
RECOVERABILITY_PATH = (
    ROOT / "docs/thesis/evidence/p3_seed50_report_pack_recoverability_20261018.json"
)
DEFAULT_OUTPUT = CAMPAIGN_DIR / "reports/findings/seed50_run_cards_20261018"
EXPECTED_FINGERPRINT = "91d2f05c4c13e67753a47b88a1c900f4887ced66dad51980a018eb09465dd10e"
EXPECTED_STATUS_COUNTS = {"failed": 4, "interrupted": 3, "succeeded": 11}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_inputs(summary: dict[str, Any], recovery: dict[str, Any]) -> None:
    if summary.get("campaign_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValueError("campaign fingerprint does not match the reconciled seed-50 screen")
    if summary.get("seed") != 50 or summary.get("n_envs") != 64:
        raise ValueError("campaign seed or environment count differs from the recorded screen")
    if summary.get("requested_steps_per_variant") != 20_000_000:
        raise ValueError("campaign budget differs from the recorded screen")
    jobs = summary.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 18:
        raise ValueError("expected all 18 campaign run records")
    keys = [job.get("run_key") for job in jobs if isinstance(job, dict)]
    if len(keys) != 18 or len(set(keys)) != 18 or any(not isinstance(key, str) for key in keys):
        raise ValueError("campaign run keys must be 18 unique strings")
    actual_status_counts = Counter(str(job.get("controller_status")) for job in jobs)
    if dict(actual_status_counts) != EXPECTED_STATUS_COUNTS:
        raise ValueError(f"unexpected controller status counts: {dict(actual_status_counts)}")
    if recovery.get("schema") != "aalto.seed50_screen_report_recoverability.v1":
        raise ValueError("unsupported report-pack recoverability report")
    if (
        recovery.get("inputs", {}).get("source_campaign_fingerprint")
        != "3474621efd6a1481bd2a20bce0cfd2c0b69969df1849c6cd738d19dc5bb4ba34"
    ):
        raise ValueError("recoverability report does not identify the source campaign lineage")
    disposition = recovery.get("checkpoint_recoverability", {}).get("disposition", "")
    if "unavailable" not in disposition:
        raise ValueError("recoverability report does not establish unavailable report packs")


def card_status(job: dict[str, Any]) -> str:
    if job.get("controller_status") == "interrupted":
        return "not-tested"
    gate = job.get("train_equity_gate")
    if isinstance(gate, dict) and gate.get("ok") is False:
        return "failed"
    metrics = job.get("train_metrics")
    test_metrics = job.get("test_metrics_report_only")
    if (
        isinstance(metrics, dict)
        and isinstance(test_metrics, dict)
        and metrics.get("n_trades") == 0
        and test_metrics.get("n_trades") == 0
    ):
        return "inactive"
    return "engineering-only"


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unavailable"
    return f"{value:.{digits}f}"


def _metrics_table(label: str, metrics: Any) -> list[str]:
    if not isinstance(metrics, dict):
        return [f"{label}: unavailable in the retained campaign summary."]
    return [
        f"- **{label}:** total PnL `{_fmt(metrics.get('total_pnl'), 2)}` USD; "
        f"return `{_fmt(metrics.get('total_return_pct'), 4)}%`; "
        f"max drawdown `{_fmt(float(metrics['max_drawdown']) * 100, 3) if isinstance(metrics.get('max_drawdown'), (int, float)) else 'unavailable'}%`; "
        f"Sharpe `{_fmt(metrics.get('sharpe'), 4)}`; "
        f"positions/trades `{metrics.get('n_trades', 'unavailable')}`; "
        f"breaches `{metrics.get('breach_count', 'unavailable')}`."
    ]


def render_card(
    job: dict[str, Any],
    *,
    summary_sha256: str,
    recovery_sha256: str,
    recovery_path: str,
) -> str:
    run_key = str(job["run_key"])
    variant = str(job["variant"])
    status = card_status(job)
    gate = job.get("train_equity_gate")
    gate_text = (
        f"`{gate.get('reason', 'unreported')}` (pass={gate.get('ok')})"
        if isinstance(gate, dict)
        else "not available"
    )
    coverage = job.get("training_episode_coverage")
    if isinstance(coverage, dict):
        exposure = (
            f"{coverage.get('full_split_episode_count', 0)} full-split episodes across "
            f"{coverage.get('environments_with_full_split_episode', 0)} environments; "
            f"telemetry status `{coverage.get('status', 'unknown')}`."
        )
    else:
        exposure = "Full-split telemetry is unavailable in the retained per-run summary."

    lines = [
        f"# Finding — {variant}, seed 50, attempt {job.get('attempt', 'unknown')}",
        "",
        f"**Status:** `{status}` — summary-derived, exploratory only; raw report pack unavailable.",
        "**Evidence role:** one-seed screen reconciliation and disposition.",
        "**Campaign:** `ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016`",
        f"**Run key:** `{run_key}`",
        f"**Campaign fingerprint:** `{EXPECTED_FINGERPRINT}`",
        "",
        "## Finding",
        "",
        f"The retained schema-v2 campaign summary reports controller status "
        f"`{job.get('controller_status', 'unknown')}`, exit code `{job.get('exit_code', 'unknown')}`, "
        f"and train-survival gate {gate_text}. Requested budget was 20,000,000 PPO steps; "
        "completed steps cannot be verified from the retained summary. The recoverability "
        "inventory found no matching report pack/checkpoint in this checkout's `outputs/` tree "
        f"and did not search external backups. Recorded attempt disposition: "
        f"{job.get('failure_reason') or 'no failure reason recorded'}.",
        "",
        f"**Training exposure record:** {exposure} This is episode telemetry reported by the "
        "campaign summary, not evidence of successful training-budget completion.",
        "",
        "## Summary-reported metrics (not canonical numerical reports)",
        "",
        "The metrics below are not a canonical numerical report; they are retained summary "
        "values whose raw report packs are unavailable.",
        "",
        *_metrics_table("Train replay", job.get("train_metrics")),
        *_metrics_table("Report-only test replay", job.get("test_metrics_report_only")),
        "",
        "The test interval was previously inspected and is exploratory/report-only. These "
        "numbers are copied from the retained campaign summary; raw equity/trade/position "
        "packages are unavailable for canonical reconstruction or independent replay.",
        "",
        "## Evidence and provenance",
        "",
        f"- Campaign summary: `outputs/gpu_campaigns/ladder18_seed50_20m_full_split_n64_cpumask_recovery_20261016/reports/campaign_results_summary_v2.json` "
        f"(SHA-256 `{summary_sha256}`).",
        f"- Report-pack recoverability audit: `{recovery_path}` (SHA-256 `{recovery_sha256}`); "
        "its scope is this checkout's `outputs/` tree, not external backups.",
        f"- Summary-recorded attempt artifact directory: `{job.get('run_artifact_dir', 'unavailable')}` "
        "(not present according to the recoverability audit).",
        "- Per-run artifact hashes are retained inside the summary but cannot be rechecked "
        "against the absent run files.",
        "",
        "## Allowed interpretation and limits",
        "",
        "- This card preserves the recorded one-seed controller disposition, episode-coverage "
        "count, and summary metrics for the named run key.",
        "- The historical screen is invalid for variant selection: model-visible lookahead was "
        "found in 17/18 configurations, and the sole profile passing that bounded poison check "
        "failed its training survival gate.",
        "- Do not use this card as a canonical numerical report, evidence of completed 20M "
        "training, a strategy-quality claim, a matched comparison, or confirmatory evidence.",
        "- This run does not establish profitability, robustness, generalization, or causal "
        "treatment effects.",
        "",
        "**Likely thesis destination:** appendix table documenting exploratory screen "
        "dispositions and unavailable artifacts, with no performance-selection claim.",
        "",
    ]
    return "\n".join(lines)


def build_package(
    *,
    summary_path: Path,
    recoverability_path: Path,
    output_dir: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    summary = read_object(summary_path)
    recovery = read_object(recoverability_path)
    validate_inputs(summary, recovery)
    summary_sha256 = sha256_file(summary_path)
    recovery_sha256 = sha256_file(recoverability_path)
    recovery_inputs = recovery.get("inputs", {})
    if recovery_inputs.get("summary_sha256") != summary_sha256:
        raise ValueError("recoverability audit summary hash does not match the input summary")
    campaign_path_text = recovery_inputs.get("campaign_path")
    campaign_sha256 = recovery_inputs.get("campaign_sha256")
    if not isinstance(campaign_path_text, str) or not isinstance(campaign_sha256, str):
        raise ValueError("recoverability audit is missing campaign identity hashes")
    campaign_path = Path(campaign_path_text)
    if not campaign_path.is_absolute():
        campaign_path = ROOT / campaign_path
    if not campaign_path.is_file() or sha256_file(campaign_path) != campaign_sha256:
        raise ValueError("campaign manifest does not match the recoverability audit hash")
    generator_sha256 = sha256_file(Path(__file__).resolve())
    recovery_relpath = display_path(recoverability_path)

    output_dir.mkdir(parents=True, exist_ok=False)
    cards: list[dict[str, str]] = []
    for job in sorted(summary["jobs"], key=lambda item: str(item["run_key"])):
        filename = f"{job['run_key']}.md"
        path = output_dir / filename
        path.write_text(
            render_card(
                job,
                summary_sha256=summary_sha256,
                recovery_sha256=recovery_sha256,
                recovery_path=recovery_relpath,
            ),
            encoding="utf-8",
        )
        cards.append(
            {
                "run_key": str(job["run_key"]),
                "variant": str(job["variant"]),
                "status": card_status(job),
                "path": filename,
                "sha256": sha256_file(path),
            }
        )

    status_counts = dict(Counter(card["status"] for card in cards))
    zero_activity_run_keys = [
        str(job["run_key"])
        for job in summary["jobs"]
        if isinstance(job.get("train_metrics"), dict)
        and isinstance(job.get("test_metrics_report_only"), dict)
        and job["train_metrics"].get("n_trades") == 0
        and job["test_metrics_report_only"].get("n_trades") == 0
    ]
    readme_lines = [
        "# Seed-50 screen finding cards — provenance-limited package",
        "",
        "These 18 cards preserve per-run dispositions and summary-derived values. They do not "
        "restore the missing raw report packs, validate those metrics, or make the screen usable "
        "for selection. The P3 recoverability audit searched this checkout's `outputs/` tree; "
        "external backups were not searched.",
        "",
        f"Campaign fingerprint: `{EXPECTED_FINGERPRINT}`.",
        f"Campaign summary SHA-256: `{summary_sha256}`.",
        f"Recoverability audit SHA-256: `{recovery_sha256}`.",
        f"Generator SHA-256: `{generator_sha256}`.",
        f"Status counts: `{json.dumps(status_counts, sort_keys=True)}`.",
        f"Summary-reported zero-trade train/test run keys ({len(zero_activity_run_keys)}): "
        f"`{', '.join(zero_activity_run_keys)}`.",
        "Status semantics: `failed` means the recorded train survival gate failed; `inactive` "
        "means a controller-succeeded run recorded zero trades in both splits; `not-tested` "
        "means the attempt was interrupted; `engineering-only` means a nonzero-activity "
        "controller-succeeded row that remains inadmissible because the screen is contaminated.",
        "",
        "| Run key | Status | Card |",
        "|---|---|---|",
    ]
    readme_lines.extend(
        f"| `{card['run_key']}` | `{card['status']}` | [{card['path']}]({card['path']}) |"
        for card in cards
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema": "aalto.seed50_screen_finding_cards.v1",
        "created_at_utc": created_at_utc or datetime.now(UTC).isoformat(),
        "evidence_role": "summary_derived_exploratory_reconciliation_not_canonical_reports",
        "campaign_fingerprint": EXPECTED_FINGERPRINT,
        "campaign_manifest": {
            "path": display_path(campaign_path),
            "sha256": campaign_sha256,
        },
        "source_summary": {
            "path": display_path(summary_path),
            "sha256": summary_sha256,
        },
        "recoverability_audit": {
            "path": recovery_relpath,
            "sha256": recovery_sha256,
        },
        "generator": {
            "path": display_path(Path(__file__)),
            "sha256": generator_sha256,
        },
        "status_counts": status_counts,
        "summary_zero_trade_train_test_run_keys": zero_activity_run_keys,
        "cards": cards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--recoverability", type=Path, default=RECOVERABILITY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_package(
        summary_path=args.summary,
        recoverability_path=args.recoverability,
        output_dir=args.out,
    )
    print(
        f"wrote {len(manifest['cards'])} summary-derived finding cards to {args.out}; "
        f"status_counts={manifest['status_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
