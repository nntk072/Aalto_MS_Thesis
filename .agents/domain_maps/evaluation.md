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
| `evaluation/rollout/model.py` | Evaluates backend-neutral decision providers through TradingEnv |
| `evaluation/rollout/provider.py` | Adapts a backend-neutral decision provider for environment rollouts |
| `evaluation/reporting/export.py` | Run artifact saving |
| `evaluation/reporting/box_table.py` | Shared evaluation/training report table renderer |
| `evaluation/reporting/diversity_artifacts.py` | Persist decoded decision-diversity reports |
| `evaluation/reporting/diversity_table.py` | Render typed diversity reports as terminal tables |
| `evaluation/metrics/decision_diversity.py` | Classify reference families and compute decision-diversity statistics |
| `evaluation/metrics/trade_pairs.py` | Pair semantic trade-open and close rows for metric, report, and chart consumers |
| `evaluation/metrics/candle_quality.py` | Identify valid/tradable OHLC rows and exact wick timestamps |
| `evaluation/reanalysis/runner.py` | Re-evaluate existing checkpoint (no retraining) |
| `simulation/environment/factory.py` | Builds evaluation environment kwargs consumed by rollout/reanalysis |
| `evaluation/metrics/trades.py` | Trade-level metrics |
| `evaluation/plotting/plots.py` | Compatibility exports for static chart families |
| `evaluation/plotting/common.py` | Shared Matplotlib setup, palette, and file output |
| `evaluation/plotting/equity.py` | Equity and drawdown charts |
| `evaluation/plotting/performance.py` | Daily PnL, distribution, and direction charts |
| `evaluation/plotting/comparison.py` | Baseline comparison charts |
| `evaluation/plotting/orders.py` | Price/order and per-trade charts |
| `evaluation/plotting/diversity.py` | Decision-diversity diagnostic charts |
| `evaluation/plotting/plots_interactive.py` | Compatibility exports for interactive chart families |
| `evaluation/plotting/interactive_common.py` | Shared Plotly availability, palette, and HTML output |
| `evaluation/plotting/interactive_performance.py` | Interactive equity, drawdown, PnL, and distribution charts |
| `evaluation/plotting/interactive_comparison.py` | Interactive baseline comparisons |
| `evaluation/plotting/interactive_orders.py` | Interactive price/order and per-trade charts |
| `evaluation/plotting/interactive_po3.py` | Interactive PO3 and FVG charts |
| `evaluation/plotting/trade_plots.py` | Per-trade plots |
| `evaluation/plotting/po3_plots.py` | PO3-specific plots |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `run_episode()` | `evaluation/rollout/runner.py` | Run one full episode, return metrics |
| `calculate_metrics()` | `evaluation/metrics/performance.py` | Compute performance metrics |
| `purged_walk_forward()` | `market_data/splits/walkforward.py` | Purged + embargoed WF splits |
| `evaluate_provider()` | `evaluation/rollout/model.py` | Walk a backend-neutral provider through TradingEnv |
| `save_run()` | `evaluation/reporting/export.py` | Save run artifacts to disk |

## Direct Dependencies

- `simulation/environment/env.py` (evaluation environment)
- `simulation/environment/factory.py` (environment construction settings)
- `trading/execution/costs.py` (cost model for evaluation)
- `decisioning/contracts.py` and provider adapters (backend-neutral inference)

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

## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add a metric | scientific metric | `evaluation/metrics/` | `MetricResult`; focused metric fixture | one metric and report wiring | training internals and renderers |
| Add a decision provider adapter | evaluation boundary | `evaluation/rollout/provider.py` | `DecisionProvider`; `tests/test_eval/test_decision_provider_rollout.py` | adapter and evaluation integration test | learner implementation |

## Axis of Change

- **Axis:** evaluation case, metric suite, provenance, and report rendering.
- **Owner:** `evaluation/`; simulation owns environment construction and state transitions.
- **Stable boundary:** evaluation request/results and versioned run provenance.
- **Expected edit radius:** one metric or renderer plus its focused test; workflow changes may touch reanalysis composition.
- **Must not depend on:** training implementation or learner-specific internals.

1. **Purged walk-forward** — `purge_bars` removed from train end, `embargo_bars` from test start
2. **Same risk params as training** — `risk_frac_range` and `rr_ratio_range` must match training env
3. **Episodic=False for RL eval** — guardrail breach blocks new trading for session, not whole rollout
4. **Training/evaluation action mode parity** — final train/test replays use
   the configured policy action mode; PPO defaults to sampled actions to match
   training. Record the mode and action RNG seed. Deterministic replay is an
   explicit sensitivity analysis and remains useful for exact replay checks.
5. **Matched comparisons** — dataset, split, feature/reward/cost/execution,
   training budget, seed, starting capital, model/environment settings, and
   evaluation window must match; previously inspected OOS remains historical.
## Verified Source Paths

- `evaluation/rollout/provider.py`
- `evaluation/reanalysis/runner.py`
- `evaluation/reanalysis/legacy_sb3.py`
- `evaluation/metrics/performance.py`
- `evaluation/contracts.py`
- `evaluation/reporting/geometry.py`
- `evaluation/reporting/dataset_renderers.py`
- `evaluation/reporting/diversity_artifacts.py`
- `evaluation/plotting/diversity.py`
