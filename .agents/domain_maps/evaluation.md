# Evaluation Domain

## Primary Files

| File | Role | Lines |
|------|------|-------|
| `quant_rl/evaluation/runner.py` | Policy-agnostic episode runner | 62 |
| `quant_rl/evaluation/metrics.py` | Performance metrics (Sharpe, drawdown, etc.) | ~150 |
| `quant_rl/evaluation/calibration.py` | Calibration analysis | ~100 |
| `quant_rl/evaluation/bootstrap_ci.py` | Bootstrap confidence intervals | ~80 |
| `quant_rl/evaluation/report.py` | Multi-seed report generation | ~100 |
| `quant_rl/evaluation/walkforward.py` | Purged + embargoed walk-forward splits | 69 |
| `quant_rl/eval/rollout.py` | RL model evaluation through TradingEnv | 152 |
| `quant_rl/eval/export.py` | Run artifact saving | ~80 |
| `quant_rl/eval/trade_metrics.py` | Trade-level metrics | ~80 |
| `quant_rl/eval/plots.py` | Static matplotlib plots | ~200 |
| `quant_rl/eval/plots_interactive.py` | Interactive plotly plots | ~150 |
| `quant_rl/eval/chart_overlays.py` | Chart overlay visualizations | ~100 |
| `quant_rl/eval/trade_plots.py` | Per-trade plots | ~100 |
| `quant_rl/eval/trade_diagnostics.py` | Trade diagnostic plots | ~100 |
| `quant_rl/eval/po3_plots.py` | PO3-specific plots | ~80 |
| `quant_rl/eval/training_plots.py` | Training curve plots | ~80 |
| `quant_rl/eval/replot_orders.py` | Order replotting utility | ~60 |
| `quant_rl/eval/chart_indicators.py` | Chart indicator overlays | ~80 |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_episode()` | `runner.py` | Run one full episode, return metrics |
| `calculate_metrics()` | `metrics.py` | Compute performance metrics |
| `PerformanceMetrics` | `metrics.py` | Metrics dataclass |
| `purged_walk_forward()` | `walkforward.py` | Purged + embargoed WF splits |
| `WFSplit` | `walkforward.py` | Walk-forward split dataclass |
| `evaluate_model()` | `rollout.py` | Walk RL model through TradingEnv |
| `make_action_fn()` | `rollout.py` | Create obs→action callable for model |
| `save_run()` | `export.py` | Save run artifacts to disk |

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
| `tests/test_eval/test_chart_overlays.py` | Chart overlays |
| `tests/test_eval/test_trade_plots.py` | Trade plots |
| `tests/test_eval/test_po3_plots.py` | PO3 plots |
| `tests/test_rollout.py` | Model rollout |
| `tests/test_trade_metrics.py` | Trade metrics |
| `tests/test_training_plots.py` | Training plots |
| `tests/test_plots_export.py` | Plot export |

## Do NOT Read For Evaluation Tasks

- `quant_rl/features/` (feature engineering — separate domain)
- `quant_rl/models/encoder.py` (encoder detail — only if model loading fails)
- `mt5_trading/` (live trading — separate domain)

## Critical Invariants

1. **Purged walk-forward** — `purge_bars` removed from train end, `embargo_bars` from test start
2. **Same risk params as training** — `risk_frac_range` and `rr_ratio_range` must match training env
3. **Episodic=False for RL eval** — guardrail breach blocks new trading for session, not whole rollout
4. **Deterministic=True** — evaluation uses deterministic policy predictions