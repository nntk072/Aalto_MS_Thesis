# Training Domain

## Primary Files

| File | Role |
|------|------|
| `quant_rl/train/train_rl.py` | Full RL training pipeline; `_STRATEGY_CONFIGS` registry |
| `quant_rl/train/run_backtest.py` | Backtest runner entry point |
| `quant_rl/train/run_baselines.py` | Baseline strategy runner |
| `quant_rl/train/callbacks.py` | Best checkpoint eval callback |
| `quant_rl/train/auxiliary_training.py` | Auxiliary prediction head training |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `main()` | `train_rl.py` | Full training pipeline entry |
| `make_env()` | `train_rl.py` | Create TradingEnv for training |
| `_strategy_from_cfg()` | `train_rl.py` | Build (strategy, reward, weight) from config |
| `_STRATEGY_CONFIGS` | `train_rl.py` | Maps `--strategy` to variant YAML |
| `BestCheckpointEvalCallback` | `callbacks.py` | Eval-based checkpoint saving |
| `AuxiliaryTrainerCallback` | `auxiliary_training.py` | Auxiliary head training callback |

## CLI flags (`train_rl.py`)

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
- `quant_rl/config/` (training config)
- `quant_rl/eval/rollout.py` (model evaluation)
- `quant_rl/evaluation/metrics.py` (performance metrics)
- `quant_rl/eval/export.py` (run artifact saving)

## Consumers (Callers)

- `scripts/train_rl.py` (CLI wrapper)
- `scripts/compare_encoders.py` (encoder comparison)

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
3. build_features(bars, cfg) → features  (cached by FEATURE_CACHE_VERSION)
4. split_train_test(bars, features, cfg) → train/test
5. make_env(train_bars, train_feat, cfg) → env
6. build_agent(env, cfg, arch, algo) → model
7. model.learn(total_timesteps=...) → trained model
8. evaluate_model(model, test_bars, test_feat) → results
9. calculate_metrics(equity, trades) → metrics
10. save_run(run_dir, results, metrics) → artifacts
```

## Config Overrides

```bash
uv run python -m quant_rl.train.train_rl --mvp --seed=42
uv run python -m quant_rl.train.train_rl --algo sac --arch gru
uv run python -m quant_rl.train.train_rl --strategy po3_ifvg
uv run python -m quant_rl.train.train_rl --config config/features_full_po3_mtf.yaml
uv run python -m quant_rl.train.train_rl --walk-forward --wf-splits 5 --purge-bars 60
```
