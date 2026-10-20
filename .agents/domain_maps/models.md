# Models Domain

## Primary Files

| File | Role |
|------|------|
| `models/encoders/common.py` | Shared input contract, account embedding and valid-window mechanics |
| `models/encoders/tcn.py` | Causal TCN family and checkpoint compatibility encoder |
| `models/encoders/lite_tcn.py` | Lightweight depthwise-separable TCN |
| `models/encoders/gru.py` | GRU sequence encoder |
| `models/encoders/transformer.py` | Causal Transformer sequence encoder |
| `models/encoders/mtf.py` | Multi-timeframe TCN encoder |
| `models/encoders/core.py` | Temporary import compatibility for archived SB3 checkpoint module paths |
| `models/auxiliary/vae.py` | Reusable variational autoencoder architecture |
| `models/base.py` | Base classes for models |
| `models/auxiliary/heads.py` | Auxiliary prediction head |
| `models/runtime/precision.py` | Encoder-specific autocast adaptation using a caller-selected dtype |

## Key Symbols

| Symbol | File | Purpose |
|--------|------|---------|
| `TCNEncoder` | `models/encoders/tcn.py` | Temporal Convolutional Network extractor |
| `GRUEncoder` | `models/encoders/gru.py` | GRU extractor |
| `TransformerEncoder` | `models/encoders/transformer.py` | Transformer extractor |
| `VAE` | `models/auxiliary/vae.py` | Variational autoencoder |
| `VAEFeatureExtractor` | `learning/rl/vae_extractor.py` | SB3 adapter around the reusable VAE |

## Contracts and Consumers

- `decisioning/observation_state.py` (stable account-vector input width)
- `decisioning/actions.py` and `decisioning/observations.py` (checkpoint schemas)
- `learning/rl/` consumes model constructors; model code has no learning-backend dependency.

## Consumers (Callers)

- `training/runner/training.py` (calls `build_agent`)
- `scripts/research/benchmarks/compare_encoders.py` (encoder comparison)
- `learning/rl/auxiliary.py` (auxiliary head rollout training)
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

`learning/rl/agent.py::build_agent(env, cfg, arch, algo, use_vae, vae, device)`:
1. Infers `F` from `env.observation_space["seq"].shape[1]`
2. Selects encoder: `tcn` (default), `gru`, `transformer`, or `vae`
3. Sets `net_arch`: SAC uses `qf`, PPO uses `vf`
4. Wraps env in `DummyVecEnv` or a sharded VecEnv
5. Returns `PPO` or `SAC` with `MultiInputPolicy`

## Observation Space Contract

```python
obs["seq"]     : float32  [batch, T, F]   (T = cfg.env.obs_window, F = n_features)
obs["account"] : float32  [batch, 6]      (normalised account state vector)
```

When `entry_state_observation` and the entry-state machine are both enabled,
four one-hot entry-state values are appended to the account vector. Strategy
overlay actions use the flag-derived `overlay-action-v2` schema from
`decisioning/actions.py`; their width and coordinate meanings depend on the
enabled direction, stop mode, TP mode, multi-TP, and simplex flags. Persist and
validate those flags and the schema version with checkpoints. Width alone is
not sufficient to recover coordinate meanings.

## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add an encoder family | neural architecture | `models/encoders/` | encoder output contract; `tests/test_models/test_encoders.py` | one encoder and its tests | learner workflow and Slurm backend |
| Change inference precision | runtime capability | `compute/environment/precision.py`, `learning/rl/agent.py`, `models/runtime/precision.py` | device capability contract; `tests/test_utils/test_device.py` | compute selects hardware policy, learning composes it with model adaptation | compute scheduler semantics |

## Axis of Change

- **Axis:** reusable neural architectures and inference-time model behavior.
- **Owner:** `models/` owns neural architecture; `learning/rl/` owns SB3 policies, fitting, and vector environments.
- **Stable boundary:** encoder/predictor input schema and checkpoint metadata.
- **Expected edit radius:** model adaptation, learning composition, and focused tests when runtime policy changes.
- **Must not depend on:** hardware discovery, campaign workflows, learning implementations, or environment runtime state internals. Learning passes the selected dtype to model-specific autocast adaptation.

Unversioned SB3 checkpoints must be loaded only with the explicit legacy
schema opt-in and a layout key taken from their original run configuration.
The `legacy4_no_exit_v1` migration is a narrowly scoped exception for the
archived 4D TCN policy: it restores that checkpoint's original encoder and
policy, then appends the structural-exit coordinate required by the current
5D base overlay schema. It does not infer missing run configuration or apply
to other action layouts. See `learning/rl/agent.py` and
`evaluation/reanalysis/runner.py`.

## Verified Source Paths

- `models/encoders/common.py`
- `models/encoders/tcn.py`
- `models/encoders/lite_tcn.py`
- `models/encoders/gru.py`
- `models/encoders/transformer.py`
- `models/encoders/mtf.py`
- `models/encoders/core.py`
- `models/auxiliary/vae.py`
- `models/base.py`
- `learning/rl/vae_extractor.py`
- `tests/test_models/test_encoders.py`
- `models/runtime/precision.py`
