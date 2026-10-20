from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.research.thesis.evidence import build_seed50_screen_finding_cards as cards


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path]:
    jobs = []
    statuses = ["failed"] * 4 + ["interrupted"] * 3 + ["succeeded"] * 11
    for index, status in enumerate(statuses):
        train_count = 0 if index < 7 or index < 15 else 5
        jobs.append(
            {
                "run_key": f"variant_{index}__seed50",
                "variant": f"variant_{index}",
                "seed": 50,
                "attempt": 1,
                "controller_status": status,
                "exit_code": 1 if status == "failed" else (143 if status == "interrupted" else 0),
                "failure_reason": "fixture" if status != "succeeded" else None,
                "run_artifact_dir": f"runs/variant_{index}",
                "train_equity_gate": {"ok": status != "failed", "reason": "fixture"},
                "train_metrics": None
                if status == "interrupted"
                else {"n_trades": train_count, "total_pnl": 1.0, "total_return_pct": 0.001},
                "test_metrics_report_only": None
                if status == "interrupted"
                else {
                    "n_trades": 0 if 7 <= index < 15 else 2,
                    "total_pnl": -1.0,
                    "total_return_pct": -0.001,
                },
                "training_episode_coverage": {
                    "status": "available",
                    "full_split_episode_count": 0 if index == 0 else 64,
                    "environments_with_full_split_episode": 0 if index == 0 else 64,
                },
            }
        )
    summary = {
        "campaign_fingerprint": cards.EXPECTED_FINGERPRINT,
        "seed": 50,
        "n_envs": 64,
        "requested_steps_per_variant": 20_000_000,
        "jobs": jobs,
    }
    recovery: dict[str, Any] = {
        "schema": "aalto.seed50_screen_report_recoverability.v1",
        "inputs": {
            "source_campaign_fingerprint": "3474621efd6a1481bd2a20bce0cfd2c0b69969df1849c6cd738d19dc5bb4ba34"
        },
        "checkpoint_recoverability": {
            "disposition": "exact report pack recovery is unavailable from outputs"
        },
    }
    summary_path = tmp_path / "summary.json"
    recovery_path = tmp_path / "recoverability.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text('{"campaign": "fixture"}\n', encoding="utf-8")
    recovery["inputs"].update(
        {
            "summary_sha256": cards.sha256_file(summary_path),
            "campaign_path": campaign_path.name,
            "campaign_sha256": cards.sha256_file(campaign_path),
        }
    )
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    return summary_path, recovery_path


def test_card_status_preserves_failure_inactivity_and_interruption() -> None:
    assert cards.card_status({"controller_status": "interrupted"}) == "not-tested"
    assert (
        cards.card_status({"controller_status": "failed", "train_equity_gate": {"ok": False}})
        == "failed"
    )
    assert (
        cards.card_status(
            {
                "controller_status": "succeeded",
                "train_equity_gate": {"ok": True},
                "train_metrics": {"n_trades": 0},
                "test_metrics_report_only": {"n_trades": 0},
            }
        )
        == "inactive"
    )
    assert (
        cards.card_status(
            {
                "controller_status": "succeeded",
                "train_equity_gate": {"ok": True},
                "train_metrics": {"n_trades": 2},
            }
        )
        == "engineering-only"
    )


def test_build_package_writes_18_limited_cards_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cards, "ROOT", tmp_path)
    summary_path, recovery_path = _fixture_inputs(tmp_path)
    output = tmp_path / "cards"

    manifest = cards.build_package(
        summary_path=summary_path,
        recoverability_path=recovery_path,
        output_dir=output,
        created_at_utc="2026-10-18T00:00:00+00:00",
    )

    assert manifest["schema"] == "aalto.seed50_screen_finding_cards.v1"
    assert len(manifest["cards"]) == 18
    assert manifest["status_counts"] == {
        "engineering-only": 3,
        "failed": 4,
        "inactive": 8,
        "not-tested": 3,
    }
    assert len(list(output.glob("variant_*__seed50.md"))) == 18
    first_card = (output / "variant_0__seed50.md").read_text(encoding="utf-8")
    assert "not a canonical numerical report" in first_card
    assert "completed steps cannot be verified" in first_card
    assert "not evidence of successful training-budget completion" in first_card
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "raw report packs" in readme

    with pytest.raises(FileExistsError):
        cards.build_package(
            summary_path=summary_path,
            recoverability_path=recovery_path,
            output_dir=output,
        )


def test_validate_inputs_fails_closed_on_wrong_campaign_fingerprint(tmp_path: Path) -> None:
    summary_path, recovery_path = _fixture_inputs(tmp_path)
    summary = cards.read_object(summary_path)
    recovery = cards.read_object(recovery_path)
    summary["campaign_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="fingerprint"):
        cards.validate_inputs(summary, recovery)
