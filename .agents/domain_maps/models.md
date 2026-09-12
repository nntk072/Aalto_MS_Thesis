# Models Domain

## Primary Files

| File | Role | Lines |
|------|------|-------|
| `quant_rl/models/agent.py` | Agent wiring: encoder + PPO/SAC construction | 163 |
| `quant_rl/models/encoder.py` | TCN/GRU/Transformer encoder implementations | ~250 |
| `quant_rl/models/vae.py` | Variational autoencoder for narrative embedding | ~150 |
| `quant_rl/models/base.py` | Base classes for models | ~60 |
| `quant_rl/models/auxiliary.py` | Auxiliary prediction head | ~80 |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `build_agent()` | `agent.py` | Build SB3 PPO/SAC with chosen encoder |
| `TCNEncoder` | `encoder.py` | Temporal Convolutional Network encoder |
| `GRUEncoder` | `encoder.py` | GRU-based encoder |
| `TransformerEncoder` | `encoder.py` | Transformer encoder |
| `BaseFeaturesExtractor` | `encoder.py` | Base class (SB3 interface) |
| `VAE` | `vae.py` | Variational autoencoder |
| `VAEFeatureExtractor` | `vae.py` | VAE as SB3 feature extractor |

## Direct Dependencies

- `quant_rl/envs/trading_env.py` (observation space inference — `build_agent` reads `env.observation_space`)

## Consumers (Callers)

- `quant_rl/train/train_rl.py` (calls `build_agent`)
- `quant_rl/scripts/compare_encoders.py` (encoder comparison)
- `quant_rl/train/auxiliary_training.py` (auxiliary head training)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_models/test_encoders.py` | Encoder implementations |
| `tests/test_models/test_sac_agent.py` | SAC agent construction |
| `tests/test_models/test_vae.py` | VAE model |
| `tests/test_models/test_auxiliary.py` | Auxiliary head |

## Do NOT Read For Model Tasks

- `quant_rl/features/` (features — separate domain)
- `quant_rl/backtest/` (backtest — separate domain)
- `quant_rl/train/` (training — orchestrates, doesn't implement models)
- `mt5_trading/` (live trading — separate domain)

## Agent Construction

`build_agent(env, cfg, arch, algo, use_vae, vae, device)`:
1. Infers `F` from `env.observation_space["seq"].shape[1]`
2. Selects encoder: `tcn` (default), `gru`, `transformer`, or `vae`
3. Sets `net_arch`: SAC uses `qf`, PPO uses `vf`
4. Wraps env in `DummyVecEnv`
5. Returns `PPO` or `SAC` with `MultiInputPolicy`

## Observation Space Contract

```python
obs["seq"]     : float32  [batch, T, F]   (T = cfg.env.obs_window, F = n_features)
obs["account"] : float32  [batch, 5]      (normalised account state vector)
```