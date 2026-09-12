# Training Domain

## Primary Files

| File | Role | Lines |
|------|------|-------|
| `quant_rl/train/train_rl.py` | Full RL training pipeline | 614 |
| `quant_rl/train/run_backtest.py` | Backtest runner entry point | ~80 |
| `quant_rl/train/run_baselines.py` | Baseline strategy runner | ~80 |
| `quant_rl/train/callbacks.py` | Best checkpoint eval callback | ~100 |
| `quant_rl/train/auxiliary_training.py` | Auxiliary prediction head training | ~100 |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `main()` | `train_rl.py` | Full training pipeline entry |
| `make_env()` | `train_rl.py` | Create TradingEnv for training |
| `BestCheckpointEvalCallback` | `callbacks.py` | Eval-based checkpoint saving |
| `AuxiliaryTrainerCallback` | `auxiliary_training.py` | Auxiliary head training callback |

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

## Do NOT Read For Training Tasks

- `quant_rl/features/indicators.py` (indicator detail — only if modifying features)
- `quant_rl/models/encoder.py` (encoder detail — only if modifying architecture)
- `mt5_trading/` (live trading — separate domain)

## Training Pipeline

```
1. load_config() → cfg
2. run_pipeline(cfg) → bars
3. build_features(bars, cfg) → features
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
python -m quant_rl.train.train_rl --mvp              # 30-day MVP
python -m quant_rl.train.train_rl --algo sac         # SAC algorithm
python -m quant_rl.train.train_rl --arch transformer # Transformer encoder
python -m quant_rl.train.train_rl --seed 42          # reproducibility
```