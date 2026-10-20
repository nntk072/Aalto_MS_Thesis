# Models Domain

## Primary Files

| File | Role |
|------|------|
| `models/factory/agent.py` | Agent wiring: encoder + PPO/SAC construction |
| `models/encoders/core.py` | TCN/GRU/Transformer encoder implementations |
| `models/auxiliary/vae.py` | Variational autoencoder (standalone training only) |
| `models/base.py` | Base classes for models |
| `models/auxiliary/heads.py` | Auxiliary prediction head |

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

- `simulation/environment/env.py` (observation space inference — `build_agent` reads `env.observation_space`)

## Consumers (Callers)

- `training/runner/training.py` (calls `build_agent`)
- `scripts/research/benchmarks/compare_encoders.py` (encoder comparison)
- `training/callbacks/auxiliary.py` (auxiliary head training)
- `live/runtime/strategy.py` (model loading for live)

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_models/test_encoders.py` | Encoder implementations |
| `tests/test_models/test_sac_agent.py` | SAC agent construction |
| `tests/test_models/test_vae.py` | VAE model |
| `tests/test_models/test_auxiliary.py` | Auxiliary head |

## Do NOT Read For Model Tasks

- `features/` (features — separate domain)
- `simulation/backtest/` (backtest — separate domain)
- `training/` (training — orchestrates, doesn't implement models)
- `integrations/mt5/` (live trading — separate domain)

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
obs["account"] : float32  [batch, 6]      (normalised account state vector)
```

When `entry_state_observation` and the entry-state machine are both enabled,
four one-hot entry-state values are appended to the account vector. Strategy
overlay actions use the flag-derived `overlay-action-v2` schema from
`simulation/contracts/actions.py`; their width and coordinate meanings depend on the
enabled direction, stop mode, TP mode, multi-TP, and simplex flags. Persist and
validate those flags and the schema version with checkpoints. Width alone is
not sufficient to recover coordinate meanings.

Unversioned SB3 checkpoints must be loaded only with the explicit legacy
schema opt-in and a layout key taken from their original run configuration.
The `legacy4_no_exit_v1` migration is a narrowly scoped exception for the
archived 4D TCN policy: it restores that checkpoint's original encoder and
policy, then appends the structural-exit coordinate required by the current
5D base overlay schema. It does not infer missing run configuration or apply
to other action layouts. See `models/factory/agent.py` and
`evaluation/reanalysis/runner.py`.
