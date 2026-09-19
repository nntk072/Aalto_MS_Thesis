# Validation Protocol (T-03.2)

## Roles of calendar slices

| Slice | Dates (default) | Allowed use |
|-------|-----------------|-------------|
| Train / IS | ≤ `data.split.train_end` | Fit policies, fit-scoped transforms |
| Walk-forward folds | Inside the train calendar | Architecture / hyperparameter **selection** |
| Locked OOS | ≥ `data.split.test_start` | **Final report only** |

Default dates live in `quant_rl/config/default.yaml` and `config/experiments.yaml`
(`train_end: 2025-12-31`, `test_start: 2026-01-01`).

## Rules

1. Do **not** tune architecture, seeds-to-keep, or feature flags by repeatedly
   reading locked-OOS Sharpe.
2. `scripts/ablation_runner.py` writes `oos_role: final_report_only` by default.
   Pass `--allow-locked-oos-for-selection` only if you knowingly rank on the
   locked holdout (recorded in the JSON).
3. Prefer `train_rl --walk-forward` (non-overlapping purged folds, T-03.1) for
   selection metrics.
4. Changing locked OOS dates is HUMAN-GATED and must update
   [threats_to_validity.md](threats_to_validity.md).

## Engine

Unless a table caption says otherwise, thesis metrics come from **`TradingEnv`**
episode evaluation (`run_episode` / `evaluate_model`), not `run_backtest`.
