"""Final evaluation artifact writers for training runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.callbacks.activity import assess_train_activity, classify_zero_trade_cause


def _write_evaluation_artifacts(
    *,
    run_dir: Path,
    seed: int,
    completed_timesteps: int,
    train_result: dict[str, Any],
    test_result: dict[str, Any],
    train_metrics: Any,
    test_metrics: Any,
    train_gate: dict[str, Any],
    train_activity_gate: dict[str, Any],
    account_currency: str,
) -> None:
    """Write an explicit train/test evaluation report with unambiguous roles."""
    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    def split_payload(
        role: str,
        result: dict[str, Any],
        metrics: Any,
        *,
        activity_status: str,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "policy_action_mode": str(result.get("policy_action_mode", "stochastic")),
            "action_rng_seed": result.get("action_rng_seed"),
            "seed": int(seed),
            "completed_training_timesteps": int(completed_timesteps),
            "account_currency": str(account_currency),
            "metrics": asdict(metrics),
            "activity_status": activity_status,
            "n_sessions": int(result.get("n_sessions", 0)),
            "days_traded": int(result.get("days_traded", 0)),
            "survived_full_year": bool(result.get("survived_full_year", False)),
            "completed_data_split": bool(result.get("completed_data_split", False)),
            "episode_end_reason": result.get("episode_end_reason"),
            "fail_time": (
                str(result["fail_time"]) if result.get("fail_time") is not None else None
            ),
            "max_trailing_dd": float(result.get("max_trailing_dd", metrics.max_drawdown)),
            "breach_events": int(result.get("n_breach_sessions", 0)),
            "entry_diagnostics": result.get("entry_diag", {}),
        }

    train_payload = split_payload(
        "in_sample_training_evaluation",
        train_result,
        train_metrics,
        activity_status=str(train_activity_gate["status"]),
    )
    test_activity = assess_train_activity(int(test_metrics.n_trades))
    test_payload = split_payload(
        "report_only_test_evaluation",
        test_result,
        test_metrics,
        activity_status=str(test_activity["status"]),
    )
    test_payload["oos_role"] = (
        "previously_inspected_holdout; report-only; not pristine confirmation"
    )
    train_payload["equity_gate"] = train_gate
    train_payload["activity_gate"] = train_activity_gate
    train_payload["zero_trade_diagnosis"] = classify_zero_trade_cause(
        int(train_metrics.n_trades),
        train_result.get("entry_state_diagnostics"),
        train_result.get("entry_diag"),
    )
    test_payload["zero_trade_diagnosis"] = classify_zero_trade_cause(
        int(test_metrics.n_trades),
        test_result.get("entry_state_diagnostics"),
        test_result.get("entry_diag"),
    )
    (evaluation_dir / "train_in_sample.json").write_text(
        json.dumps(train_payload, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )
    (evaluation_dir / "test_report_only.json").write_text(
        json.dumps(test_payload, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "seed": int(seed),
        "completed_training_timesteps": int(completed_timesteps),
        "training_in_sample": train_payload,
        "test_report_only": test_payload,
        "train_equity_gate": train_gate,
        "train_activity_gate": train_activity_gate,
        "train_survival_ok": bool(train_gate.get("ok")),
        "train_activity_ok": bool(train_activity_gate.get("ok")),
        "training_admissible": bool(train_gate.get("ok")) and bool(train_activity_gate.get("ok")),
        "zero_trade_diagnosis": train_payload["zero_trade_diagnosis"],
    }
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, allow_nan=True), encoding="utf-8"
    )

    def line(label: str, payload: dict[str, Any]) -> str:
        metrics = payload["metrics"]
        return (
            f"| {label} | {metrics['total_return_pct']:+.2f}% | "
            f"{metrics['total_pnl']:+.2f} {account_currency} | "
            f"{metrics['max_drawdown']:.2%} | "
            f"{metrics['sharpe']:.3f} | {metrics['profit_factor']:.3f} | "
            f"{metrics['win_rate']:.1%} | {metrics['n_trades']} |"
        )

    report = [
        "# Evaluation results",
        "",
        f"Seed: {seed}  ",
        f"Training steps completed: {completed_timesteps:,}",
        "",
        "The dashboard's periodic train-prefix monitor is a progress check. These are the two final replays.",
        (
            "Policy action mode: "
            f"train={train_payload['policy_action_mode']}, "
            f"test={test_payload['policy_action_mode']}; "
            f"action RNG seeds: train={train_payload['action_rng_seed']}, "
            f"test={test_payload['action_rng_seed']}."
        ),
        (
            "Both final replays use stochastic policy actions, matching PPO training."
            if train_payload["policy_action_mode"] == "stochastic"
            and test_payload["policy_action_mode"] == "stochastic"
            else "Final replay action mode is recorded separately for each split."
        ),
        "",
        f"| Evaluation | Return | Net PnL ({account_currency}) | Max drawdown | Sharpe | Profit factor | Win rate | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        line("Train (in-sample)", train_payload),
        line("Test (report-only; previously inspected)", test_payload),
        "",
        f"Train equity gate: {'PASS' if train_gate.get('ok') else 'FAIL'}",
        f"Train activity gate: {train_activity_gate['status']} "
        f"(trades={train_activity_gate['n_trades']})",
        f"Zero-trade diagnosis: {train_payload['zero_trade_diagnosis'] or 'none'}",
        "",
        "The test split is report-only and must not be described as pristine confirmatory evidence.",
        "",
    ]
    (evaluation_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
