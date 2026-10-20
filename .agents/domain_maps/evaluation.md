# Evaluation Domain

One canonical `evaluation/` package owns rollout, metrics, reanalysis, reporting, and plotting.
Purged walk-forward split construction lives in `market_data/splits/walkforward.py`.

## Primary Files

| File | Role |
|------|------|
| `evaluation/rollout/runner.py` | Policy-agnostic episode runner |
| `evaluation/metrics/performance.py` | Performance metrics (Sharpe, drawdown, etc.) |
| `evaluation/metrics/calibration.py` | Calibration analysis |
| `evaluation/metrics/bootstrap.py` | Bootstrap confidence intervals |
| `evaluation/reporting/summary.py` | Multi-seed report generation |
| `market_data/splits/walkforward.py` | Purged + embargoed walk-forward splits |
| `evaluation/metrics/matched_comparison.py` | Timestamp-paired treatment/control intervals with fail-closed provenance |
| `evaluation/reporting/run_manifest.py` | Per-trial dataset, source, config, cache and checkpoint provenance |
| `evaluation/rollout/model.py` | RL model evaluation through TradingEnv |
| `evaluation/reporting/export.py` | Run artifact saving |
| `evaluation/reanalysis/runner.py` | Re-evaluate existing checkpoint (no retraining) |
| `evaluation/metrics/trades.py` | Trade-level metrics |
| `evaluation/plotting/plots.py` | Static matplotlib plots |
| `evaluation/plotting/plots_interactive.py` | Interactive plotly plots |
| `evaluation/plotting/trade_plots.py` | Per-trade plots |
| `evaluation/plotting/po3_plots.py` | PO3-specific plots |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_episode()` | `evaluation/rollout/runner.py` | Run one full episode, return metrics |
| `calculate_metrics()` | `evaluation/metrics/performance.py` | Compute performance metrics |
| `purged_walk_forward()` | `market_data/splits/walkforward.py` | Purged + embargoed WF splits |
| `evaluate_model()` | `evaluation/rollout/model.py` | Walk RL model through TradingEnv |
| `save_run()` | `evaluation/reporting/export.py` | Save run artifacts to disk |

## Direct Dependencies

- `simulation/environment/env.py` (evaluation environment)
- `trading/execution/costs.py` (cost model for evaluation)
- `models/factory/agent.py` (model loading for evaluation)

## Consumers (Callers)

- `training/runner/training.py` (evaluates after training)
- `scripts/research/thesis/evaluation/report_g3.py` (report generation)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_evaluation/test_runner.py` | Episode runner |
| `tests/test_evaluation/test_metrics.py` | Metrics calculation |
| `tests/test_evaluation/test_calibration_ci.py` | Calibration |
| `tests/test_evaluation/test_matched_comparison_unittest.py` | Matched-run fields, timestamp windows and paired intervals |
| `tests/test_eval/test_run_manifest.py` | Run-manifest identity and fail-closed artifact requirements |
| `tests/test_evaluation/test_report.py` | Report generation |
| `tests/test_eval/test_walkforward.py` | Walk-forward splits |
| `tests/test_eval/test_evaluate_metrics_contract.py` | Metrics contract |
| `tests/test_eval/test_trade_plots.py` | Trade plots |
| `tests/test_eval/test_po3_plots.py` | PO3 plots |
| `tests/test_rollout.py` | Model rollout |
| `tests/test_trade_metrics.py` | Trade metrics |

## Do NOT Read For Evaluation Tasks

- `features/technical/indicators.py` (feature implementation detail)
- `models/encoders/core.py` (encoder detail — only if model loading fails)
- `integrations/mt5/` (live trading — separate domain)

## Critical Invariants

1. **Purged walk-forward** — `purge_bars` removed from train end, `embargo_bars` from test start
2. **Same risk params as training** — `risk_frac_range` and `rr_ratio_range` must match training env
3. **Episodic=False for RL eval** — guardrail breach blocks new trading for session, not whole rollout
4. **Deterministic=True** — evaluation uses deterministic policy predictions
5. **Matched comparisons** — dataset, split, feature/reward/cost/execution,
   training budget, seed, starting capital, model/environment settings, and
   evaluation window must match; previously inspected OOS remains historical.
