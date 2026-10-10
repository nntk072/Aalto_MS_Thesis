# Training, evaluation and reproducibility

## Historical evidence records

The dated [2026-09-09 baseline runbook](../baseline/runbook_20260909T224227Z.md)
preserves its original environment, test results and provenance as a snapshot,
not current CI evidence. The [GPU campaign](../docs/training/triton_gpu_training_agent_plan.md)
and [RAM optimization](../docs/training/triton_ram_optimization_plan.md) plans contain
run-specific scheduling, resource and parity requirements; they are
historical planning records, not evidence that the proposed experiments ran.

**Code baseline:** `main@76684a3` (2026-10-10). Always record actual HEAD, dataset
identity, merged config and run artifacts before reproducing a study.

## Data-to-model execution

1. Prepare M1 source data with `python scripts/data/prepare_data.py --help` and
   inspect the expected input paths in `quant_rl/config/default.yaml`. The data
   loader, cleaning, resampling and session code is in `quant_rl/data/`.
2. `quant_rl/features/build.py:build_features` computes the selected indicators
   and structure/PO3 state. `quant_rl/data/align.py:align_timeframes` shifts
   higher-timeframe values by one completed candle by default before forward fill.
   The exception `completed_bars=False` is for callers that have already shifted;
   audit each caller when checking causality. SMT pairs are joined with a backward
   `merge_asof` in `join_symbols`.
3. Date-based training/test isolation is implemented in
   `quant_rl/data/split.py:split_train_test`. A separate `train_mask` must be
   passed to feature normalization where appropriate; slicing after fitting
   global normalization is not an equivalent safeguard. Walk-forward utilities
   are in `quant_rl/evaluation/walkforward.py`.
4. `quant_rl/train/train_rl.py:parse_train_args` exposes `--config`, `--variant`,
   `--strategy`, `--seed`, `--seeds`, `--algo`, `--arch`, `--reward`, `--mvp`,
   `--walk-forward`, `--wf-splits`, `--purge-bars` and `--embargo-bars`.
   `quant_rl/models/agent.py:build_agent` chooses SB3 PPO or SAC and TCN,
   GRU, Transformer or MTF encoder. An available path or CLI option does not
   prove every combination was validated experimentally.
5. Evaluation is implemented under `quant_rl/evaluation/` and `quant_rl/eval/`.
   Examine `quant_rl/eval/eval_run.py` and `scripts/eval/test_oos.py` for
   checkpoint loading and OOS output; use the [validation protocol](thesis/validation_protocol.md)
   for locked holdout policy.

## Safe inspection commands

These commands inspect capabilities without starting model training:

```bash
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
git rev-parse HEAD
git status --short --branch
.venv-x86/bin/python -m quant_rl.train.train_rl --help
.venv-x86/bin/python scripts/data/prepare_data.py --help
.venv-x86/bin/python -m quant_rl.eval.eval_run --help
```

Use the aarch64 environment `.venv` when on a GH200; `.venv-x86` is for x86_64.
The SSH login node need not have the CUDA environment used for GPU training.
See [operational commands](operations/RUNNING_COMMANDS.md) and
[Triton allocation policy](../scripts/triton/README.md).

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
