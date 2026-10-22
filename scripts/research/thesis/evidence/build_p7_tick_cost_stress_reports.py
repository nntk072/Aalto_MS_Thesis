#!/usr/bin/env python3
"""Build paired-seed P7 reports for 5- versus 10-point tick fills."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.thesis.evidence.analyze_paired_seed_results import (
    analyze_paired_seed_metrics,
)

CAMPAIGN_ID = "rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261021_scratch1"
CAMPAIGN_FINGERPRINT = "10e75ab7a25a53b3d4444098185fe45603811f09f2d556265c9f80ca402388ab"
REPORT_DIR = "reports/p7_tick_cost_stress_20261022"
SEEDS = (101, 102, 103, 104, 105)
COSTS = {"baseline_5pt": 5.0, "stress_10pt": 10.0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_cost_hash(training_cost_hash: str, slippage_points: float) -> str:
    content = json.dumps(
        {
            "training_costs_sha256": training_cost_hash,
            "evaluation_tick_fill_slippage_points": slippage_points,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _run_record(
    repo: Path,
    ledger: dict[str, Any],
    arm_name: str,
    seed: int,
    cost_name: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in ledger["results"]
        if item["training_arm"] == arm_name
        and int(item["seed"]) == seed
        and item["cost_level"] == cost_name
        and int(item["repetition"]) == 1
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one replay record for {cost_name}/{arm_name}/seed{seed}")
    record = matches[0]
    manifest = (repo / record["reanalysis_manifest"]).resolve()
    metrics = (repo / record["metrics_path"]).resolve()
    if _sha256(manifest) != record["reanalysis_manifest_sha256"]:
        raise ValueError(f"reanalysis manifest hash changed: {manifest}")
    if _sha256(metrics) != record["metrics_sha256"]:
        raise ValueError(f"reanalysis metrics hash changed: {metrics}")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    override = manifest_data.get("execution_cost_override", {})
    expected_points = COSTS[cost_name]
    if (
        manifest_data.get("checkpoint_sha256") != record["checkpoint_sha256"]
        or float(override.get("effective_tick_fill_slippage_points", -1)) != expected_points
        or not override.get("position_sizing_cost_model_unchanged")
        or not override.get("trigger_quote_unchanged")
    ):
        raise ValueError(f"reanalysis execution contract mismatch: {manifest}")
    return {
        **record,
        "manifest_data": manifest_data,
        "metrics_data": json.loads(metrics.read_text()),
    }


def _build_pair_payload(
    repo: Path,
    campaign_root: Path,
    source_payload: dict[str, Any],
    ledger: dict[str, Any],
    training_arm: str,
) -> dict[str, Any]:
    source_pairs = {int(pair["seed"]): pair for pair in source_payload["pairs"]}
    pairs = []
    for seed in SEEDS:
        source_arm = source_pairs[seed][training_arm]
        original_provenance = source_arm["provenance"]
        expected_checkpoint = source_arm["checkpoint_sha256"]
        outputs = {name: _run_record(repo, ledger, training_arm, seed, name) for name in COSTS}
        baseline = outputs["baseline_5pt"]
        stress = outputs["stress_10pt"]
        if (
            baseline["checkpoint_sha256"] != expected_checkpoint
            or stress["checkpoint_sha256"] != expected_checkpoint
        ):
            raise ValueError(
                f"seed {seed} {training_arm}: cost runs do not use the frozen checkpoint"
            )

        arms: dict[str, Any] = {}
        for role, cost_name, run in (
            ("control", "baseline_5pt", baseline),
            ("treatment", "stress_10pt", stress),
        ):
            metrics = run["metrics_data"]
            run_id = f"{run['training_arm']}-seed{seed}-{cost_name}"
            provenance = dict(original_provenance)
            provenance["run_id"] = run_id
            provenance["costs_sha256"] = _effective_cost_hash(
                str(original_provenance["costs_sha256"]), COSTS[cost_name]
            )
            provenance["effective_tick_fill_slippage_points"] = COSTS[cost_name]
            provenance["source_checkpoint_sha256"] = expected_checkpoint
            if provenance["seed"] != seed:
                raise ValueError(f"seed {seed}: original P7 provenance is inconsistent")
            artifact_paths = [
                run["reanalysis_manifest"],
                run["metrics_path"],
                str(Path(run["source_run"]) / "run_manifest.json"),
                run["checkpoint"],
            ]
            artifacts = []
            for artifact_path in artifact_paths:
                absolute = (
                    (repo / artifact_path).resolve()
                    if not Path(artifact_path).is_absolute()
                    else Path(artifact_path)
                )
                artifacts.append(
                    {
                        "path": str(absolute.relative_to(repo)),
                        "sha256": _sha256(absolute),
                        "size_bytes": absolute.stat().st_size,
                    }
                )
            arms[role] = {
                "run_id": run_id,
                "status": "succeeded",
                "evidence_role": "matched_exploratory",
                "provenance": provenance,
                "metrics": {"net_return_pct": float(metrics["total_return_pct"])},
                "trade_count": int(metrics["n_trades"]),
                "checkpoint_sha256": expected_checkpoint,
                "artifact_sources": artifacts,
            }
        pairs.append({"seed": seed, **arms})

    return {
        "schema": "aalto.paired_seed_metrics.v1",
        "comparison_id": f"p7_{training_arm}_tick_cost_5pt_vs_10pt_seed101_105",
        "campaign_id": CAMPAIGN_ID,
        "campaign_fingerprint": CAMPAIGN_FINGERPRINT,
        "training_arm": training_arm,
        "evidence_role": "matched_exploratory",
        "evaluation_exposure": "exposed_report_only",
        "intervention_field": "costs_sha256",
        "expected_seeds": list(SEEDS),
        "metrics": ["net_return_pct"],
        "pairs": pairs,
        "source_artifacts": [
            {
                "path": str((campaign_root / "campaign.json").relative_to(repo)),
                "sha256": _sha256(campaign_root / "campaign.json"),
            },
            {
                "path": str(
                    (
                        campaign_root
                        / "reports/p7_paired_seed_metrics_exploratory_20261022_v2.json"
                    ).relative_to(repo)
                ),
                "sha256": _sha256(
                    campaign_root / "reports/p7_paired_seed_metrics_exploratory_20261022_v2.json"
                ),
            },
            {
                "path": str((campaign_root / REPORT_DIR / "replay_ledger.json").relative_to(repo)),
                "sha256": _sha256(campaign_root / REPORT_DIR / "replay_ledger.json"),
            },
        ],
        "selection_note": (
            "Exploratory cost sensitivity on the previously inspected 2026 report interval. "
            "Each comparison holds the trained arm and seed fixed; only evaluation tick-fill "
            "slippage changes from 5.0 to 10.0 points."
        ),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[4]
    campaign_root = repo / "outputs/gpu_campaigns" / CAMPAIGN_ID
    source_path = campaign_root / "reports/p7_paired_seed_metrics_exploratory_20261022_v2.json"
    ledger_path = campaign_root / REPORT_DIR / "replay_ledger.json"
    campaign = json.loads((campaign_root / "campaign.json").read_text(encoding="utf-8"))
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if campaign.get("fingerprint") != CAMPAIGN_FINGERPRINT:
        raise ValueError("P7 campaign fingerprint changed")
    if (
        ledger.get("campaign_fingerprint") != CAMPAIGN_FINGERPRINT
        or len(ledger.get("results", [])) != 24
    ):
        raise ValueError("P7 cost-stress replay ledger is incomplete or mismatched")
    if len(ledger.get("reproducibility_checks", {})) != 4:
        raise ValueError("P7 cost-stress exact repeat checks are incomplete")

    reports: dict[str, dict[str, Any]] = {}
    for training_arm in ("control", "treatment"):
        payload = _build_pair_payload(repo, campaign_root, source_payload, ledger, training_arm)
        output_stem = f"p7_cost_stress_{training_arm}_seed101_105_20261022"
        payload_path = campaign_root / REPORT_DIR / f"{output_stem}_metrics.json"
        stats_path = campaign_root / REPORT_DIR / f"{output_stem}_statistics.json"
        for output in (payload_path, stats_path):
            if output.exists():
                raise FileExistsError(f"refusing to overwrite existing report: {output}")
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stats = analyze_paired_seed_metrics(payload, rng_seed=101, bootstrap_draws=10_000)
        stats["input_path"] = str(payload_path.resolve())
        stats["input_sha256"] = _sha256(payload_path)
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
        reports[training_arm] = {"payload": payload, "statistics": stats}

    lines = [
        "# P7 tick-fill cost sensitivity",
        "",
        "Exploratory paired-seed replay on the previously inspected 2026 report interval.",
        "Each arm and seed uses the same checkpoint, data, execution seed, action mode, and",
        "latency protocol at both costs. The only intervention is effective tick-side fill",
        "slippage: 5.0-point baseline versus 10.0-point adverse stress.",
        "",
        "The cost override changes executed prices after tick selection. It does not change",
        "tick timestamps, latency draws, protective-level trigger selection, opposite quotes,",
        "or position sizing reserves. Exact repeated seed-101 artifacts were hash-compared",
        "for both trained arms at both cost levels.",
    ]
    for training_arm, report in reports.items():
        lines.extend(["", f"## Trained arm: {training_arm}", ""])
        metric = report["statistics"]["metrics"]["net_return_pct"]
        lines.extend(
            [
                f"Mean stress-minus-baseline return: {metric['mean_treatment_minus_control']:.4f} pp.",
                "Student-t 95% CI: "
                f"[{metric['paired_student_t_ci_95pct'][0]:.4f}, "
                f"{metric['paired_student_t_ci_95pct'][1]:.4f}] pp; "
                f"paired exact sign-flip p={metric['sign_flip_p_value']:.4f}.",
                "",
                "| Seed | 5-point return (%) | 10-point return (%) | Difference (pp) | Trades at 5 / 10 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for pair in report["payload"]["pairs"]:
            baseline = pair["control"]
            stress = pair["treatment"]
            baseline_return = baseline["metrics"]["net_return_pct"]
            stress_return = stress["metrics"]["net_return_pct"]
            lines.append(
                f"| {pair['seed']} | {baseline_return:.6f} | {stress_return:.6f} | "
                f"{stress_return - baseline_return:+.6f} | "
                f"{baseline['trade_count']} / {stress['trade_count']} |"
            )
        lines.append("")
        lines.append(
            "These historical exposed results estimate sensitivity for this selected checkpoint "
            "set; they are not fresh confirmatory OOS evidence."
        )

    report_path = campaign_root / REPORT_DIR / "p7_tick_cost_stress_results_20261022.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote P7 cost-stress reports under {report_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
