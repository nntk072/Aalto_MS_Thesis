# Training Domain

## Primary Files

| File | Role |
|------|------|
| `training/runner/training.py` | Full RL training pipeline; `_STRATEGY_CONFIGS` registry |
| `training/runner/backtest.py` | Backtest runner entry point |
| `training/runner/baselines.py` | Baseline strategy runner |
| `training/callbacks/core.py` | Best checkpoint eval callback |
| `training/callbacks/auxiliary.py` | Auxiliary prediction head training |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `main()` | `training/runner/training.py` | Full training pipeline entry |
| `make_env()` | `training/runner/training.py` | Create TradingEnv for training |
| `_strategy_from_cfg()` | `training/runner/training.py` | Build (strategy, reward, weight) from config |
| `_STRATEGY_CONFIGS` | `training/runner/training.py` | Maps `--strategy` to variant YAML |
| `BestCheckpointEvalCallback` | `training/callbacks/core.py` | Eval-based checkpoint saving |
| `AuxiliaryTrainerCallback` | `training/callbacks/auxiliary.py` | Auxiliary head training callback |

## CLI flags (`training/runner/training.py`)

| Flag | Purpose |
|------|---------|
| `--mvp` | 30-day smoke test |
| `--force` | Rebuild data/feature caches |
| `--algo ppo\|sac` | RL algorithm |
| `--arch tcn\|gru\|transformer` | Encoder architecture |
| `--reward dsr\|sweep` | Reward function |
| `--strategy baseline\|po3_ifvg\|distribution` | Strategy overlay |
| `--config PATH` | Base or feature variant YAML |
| `--walk-forward` | Purged walk-forward validation |
| `--wf-splits`, `--purge-bars`, `--embargo-bars` | Walk-forward params |
| `--wandb` | Weights & Biases logging |

## Direct Dependencies

- ALL domains above (data, features, envs, backtest, models)
- `config/` (training config)
- `evaluation/rollout/model.py` (model evaluation)
- `evaluation/metrics/performance.py` (performance metrics)
- `evaluation/reporting/export.py` (run artifact saving)

## Consumers (Callers)

- `thesis-train` console entry point (`training/cli.py`)
- `scripts/research/benchmarks/compare_encoders.py` (encoder comparison)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_train/test_training_smoke.py` | Training smoke test |
| `tests/test_train/test_sac_smoke.py` | SAC training smoke |
| `tests/test_train/test_train_rl_decomposition.py` | Training decomposition |
| `tests/test_train/test_baseline_policies.py` | Baseline policies |
| `tests/test_train/test_best_checkpoint_callback.py` | Checkpoint callback |
| `tests/test_train/test_auxiliary_training.py` | Auxiliary training |
| `tests/test_train/test_seed_reproducibility.py` | Seed reproducibility |

## Training Pipeline

```
1. load_config() → cfg
2. run_pipeline(cfg) → bars
3. build_features(bars, cfg, train_mask) → features (fit scope enters the cache identity)
4. split_train_test(bars, features, cfg) → train/test
5. make_env(train_bars, train_feat, cfg) → env
6. build_agent(env, cfg, arch, algo) → model
7. model.learn(total_timesteps=...) → trained model
8. evaluate_model(model, test_bars, test_feat) → results
9. calculate_metrics(equity, trades) → metrics
10. save_run(run_dir, results, metrics) → artifacts
```

Walk-forward folds rebuild features through `build_walkforward_fold_features`:
causal bars through the fold validation end provide indicator context, while
`frozen_fold` normalization is fitted only on that fold's training indices.
Per-seed training manifests record exact train/test bar and feature hashes,
split boundaries, action schema, feature-cache identity, checkpoint digest,
runtime, and execution/metric protocols. These records are emitted by the
training path and are covered by helper tests; no full campaign artifact is
implied by the documentation.

## Config Overrides

```bash
uv run thesis-train train --mvp --seed=42
uv run thesis-train train --algo sac --arch gru
uv run thesis-train train --strategy po3_ifvg
uv run thesis-train train --config config/features/full_po3_mtf.yaml
uv run thesis-train train --walk-forward --wf-splits 5 --purge-bars 60
```
