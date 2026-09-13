# Evaluation Domain

Two packages: `quant_rl/evaluation/` (metrics, walk-forward, reporting) and
`quant_rl/eval/` (rollout, plots, export, `eval_run`).

## Primary Files

| File | Role |
|------|------|
| `quant_rl/evaluation/runner.py` | Policy-agnostic episode runner |
| `quant_rl/evaluation/metrics.py` | Performance metrics (Sharpe, drawdown, etc.) |
| `quant_rl/evaluation/calibration.py` | Calibration analysis |
| `quant_rl/evaluation/bootstrap_ci.py` | Bootstrap confidence intervals |
| `quant_rl/evaluation/report.py` | Multi-seed report generation |
| `quant_rl/evaluation/walkforward.py` | Purged + embargoed walk-forward splits |
| `quant_rl/eval/rollout.py` | RL model evaluation through TradingEnv |
| `quant_rl/eval/export.py` | Run artifact saving |
| `quant_rl/eval/eval_run.py` | Re-evaluate existing checkpoint (no retraining) |
| `quant_rl/eval/trade_metrics.py` | Trade-level metrics |
| `quant_rl/eval/plots.py` | Static matplotlib plots |
| `quant_rl/eval/plots_interactive.py` | Interactive plotly plots |
| `quant_rl/eval/trade_plots.py` | Per-trade plots |
| `quant_rl/eval/po3_plots.py` | PO3-specific plots |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_episode()` | `evaluation/runner.py` | Run one full episode, return metrics |
| `calculate_metrics()` | `evaluation/metrics.py` | Compute performance metrics |
| `purged_walk_forward()` | `evaluation/walkforward.py` | Purged + embargoed WF splits |
| `evaluate_model()` | `eval/rollout.py` | Walk RL model through TradingEnv |
| `save_run()` | `eval/export.py` | Save run artifacts to disk |

## Direct Dependencies

- `quant_rl/envs/trading_env.py` (evaluation environment)
- `quant_rl/backtest/costs.py` (cost model for evaluation)
- `quant_rl/models/agent.py` (model loading for evaluation)

## Consumers (Callers)

- `quant_rl/train/train_rl.py` (evaluates after training)
- `scripts/report_g3.py` (report generation)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_evaluation/test_runner.py` | Episode runner |
| `tests/test_evaluation/test_metrics.py` | Metrics calculation |
| `tests/test_evaluation/test_calibration_ci.py` | Calibration |
| `tests/test_evaluation/test_report.py` | Report generation |
| `tests/test_eval/test_walkforward.py` | Walk-forward splits |
| `tests/test_eval/test_evaluate_metrics_contract.py` | Metrics contract |
| `tests/test_eval/test_trade_plots.py` | Trade plots |
| `tests/test_eval/test_po3_plots.py` | PO3 plots |
| `tests/test_rollout.py` | Model rollout |
| `tests/test_trade_metrics.py` | Trade metrics |

## Do NOT Read For Evaluation Tasks

- `quant_rl/features/indicators.py` (feature implementation detail)
- `quant_rl/models/encoder.py` (encoder detail — only if model loading fails)
- `mt5_trading/` (live trading — separate domain)

## Critical Invariants

1. **Purged walk-forward** — `purge_bars` removed from train end, `embargo_bars` from test start
2. **Same risk params as training** — `risk_frac_range` and `rr_ratio_range` must match training env
3. **Episodic=False for RL eval** — guardrail breach blocks new trading for session, not whole rollout
4. **Deterministic=True** — evaluation uses deterministic policy predictions
