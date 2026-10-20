# Training, evaluation and reproducibility

## Historical evidence records

The dated [2026-09-09 baseline runbook](../baseline/runbook_20260909T224227Z.md)
preserves its original environment, test results and provenance as a snapshot,
not current CI evidence. The [GPU campaign](../docs/training/triton_gpu_training_agent_plan.md)
and [RAM optimization](../docs/training/triton_ram_optimization_plan.md) plans contain
run-specific scheduling, resource and parity requirements; they are
historical planning records, not evidence that the proposed experiments ran.

The active operational scope and current split are documented in the
[thesis execution plan](thesis/master_project_execution_plan.md). Record the
actual source revision, split inputs, resolved config, and run artifacts for
each result; dated historical planning notes do not define the current run.

## Data-to-model execution

1. Prepare M1 source data with `python market_data/commands/prepare.py --help` and
   inspect the expected input paths in `config/base/default.yaml`. The data
   loader, cleaning, resampling and session code is in `market_data/`.
2. `features/pipeline/builder.py:build_features` computes the selected indicators
   and structure/PO3 state. `market_data/transforms/align.py:align_timeframes` shifts
   higher-timeframe values by one completed candle by default before forward fill.
   The exception `completed_bars=False` is for callers that have already shifted;
   audit each caller when checking causality. SMT pairs are joined with a backward
   `merge_asof` in `join_symbols`.
3. Date-based training/test isolation is implemented in
   `market_data/splits/definitions.py:split_train_test`. A separate `train_mask` must be
   passed to feature normalization where appropriate; slicing after fitting
   global normalization is not an equivalent safeguard. Walk-forward utilities
   are in `market_data/splits/walkforward.py`.
4. `training/runner/training.py:parse_train_args` exposes `--config`, `--variant`,
   `--strategy`, `--seed`, `--seeds`, `--algo`, `--arch`, `--reward`, `--mvp`,
   `--walk-forward`, `--wf-splits`, `--purge-bars` and `--embargo-bars`.
   `models/factory/agent.py:build_agent` chooses SB3 PPO or SAC and TCN,
   GRU, Transformer or MTF encoder. An available path or CLI option does not
   prove every combination was validated experimentally.
5. Evaluation is implemented under `evaluation/` and `evaluation/`.
   Examine `evaluation/reanalysis/runner.py` and `scripts/research/thesis/evaluation/test_oos.py` for
   checkpoint loading and OOS output; use the [validation protocol](thesis/validation_protocol.md)
   for locked holdout policy.

## Safe inspection commands

These commands inspect capabilities without starting model training:

```bash
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
git rev-parse HEAD
git status --short --branch
.venv-x86/bin/thesis-train train --help
.venv-x86/bin/python market_data/commands/prepare.py --help
.venv-x86/bin/thesis-evaluate reanalyze --help
```

Use the aarch64 environment `.venv` when on a GH200; `.venv-x86` is for x86_64.
The SSH login node need not have the CUDA environment used for GPU training.
See [operational commands](operations/RUNNING_COMMANDS.md) and
[Triton allocation policy](../docs/operations/triton/README.md).

## Artifact evidence ledger (mandatory for thesis tables)

For *every* reported number, save `git_sha`, resolved config YAML, input data
hash or identity, train/test timestamps, seed, architecture, algorithm,
checkpoint path and digest, evaluation script version/arguments, costs,
account limits, trade log and metric output. Record whether the run completed
the requested training steps and whether the checkpoint is nonempty and loadable.
Distinguish train, validation, locked OOS, per-seed and aggregated metrics.
Sharpe, max drawdown, returns, trades and win rate must come from the same
evaluation protocol. Any untraceable figure is **UNVERIFIED** and must be
excluded from claims. See [historical eval pack](thesis/eval_pack.md).

## Documented research limitations

The backtest simulates a broker, fills, position management and guardrails;
it does not establish live trade execution parity. Model validity also depends
on causal higher-TF features, cost realism, split design, seed variability and
the limited period of market observations. PPO and SAC are implemented; do not
claim CPO or PPO-Lagrangian enforcement unless separately verified in code.
See [threats to validity](thesis/threats_to_validity.md) for thesis framing.
