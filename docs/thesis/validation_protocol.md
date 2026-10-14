# Validation Protocol (T-03.2)

## Roles of calendar slices

| Slice | Dates (default) | Allowed use |
|-------|-----------------|-------------|
| Train / IS | ≤ `data.split.train_end` | Fit policies, fit-scoped transforms |
| Walk-forward folds | Inside the train calendar | Architecture / hyperparameter **selection** |
| Exposed development period | 2026-01-01 through available 2026-07-03 data | Exploratory tuning only with explicit selection provenance; non-confirmatory |
| Fresh locked OOS | Newly acquired data strictly after the exposed period | **Final report only** |

Default dates live in `quant_rl/config/default.yaml` and `config/experiments.yaml`
(`train_end: 2025-12-31`, `test_start: 2026-01-01`).

## Rules

1. Do **not** tune architecture, seeds-to-keep, or feature flags by repeatedly
    reading the fresh locked holdout.
2. The researcher explicitly authorized continued development use of the
    previously inspected 2026 period on 2026-10-13. Any tuning on it is
    exploratory: permanently classify it as exposed development data, record
    `oos_role: selection_acknowledged`, and make no confirmatory or pristine
    holdout claim from it. Do not use the same period again as final OOS.
3. `scripts/matrix/ablation_runner.py` writes `oos_role: final_report_only` by default.
    Pass `--allow-locked-oos-for-selection` only for explicitly authorized
    exploratory selection on exposed data; record the selection acknowledgment.
4. `train_rl.py` requires `--allow-locked-oos-for-selection` when used with
    `--wandb` so the sweep does not log locked-OOS Sharpe by default.
    The bundled `config/wandb_sweep.yaml` opts in explicitly.
5. Prefer `train_rl --walk-forward` (non-overlapping purged folds, T-03.1) for
    selection metrics.
6. Changing locked OOS dates is HUMAN-GATED and must update
    [threats_to_validity.md](threats_to_validity.md).
7. Any OOS period whose outcomes have already been inspected during model or
    implementation decisions is historical evidence, not an untouched holdout.
    A later rerun on the same dates does not restore pristine holdout status;
    record its prior exposure in the run manifest and report.
8. The configured 2026 period was previously inspected and is now explicitly
    designated exposed development data. Available M1 and tick inputs end on
    2026-07-03. Use newly acquired and frozen data after that boundary for a
    genuinely fresh final test; until then, no final-holdout result exists.

## Engine

Unless a table caption says otherwise, thesis metrics come from **`TradingEnv`**
episode evaluation (`run_episode` / `evaluate_model`), not `run_backtest`.

For provenance and checkpoint validity, see [reproducibility](../reproducibility.md).
