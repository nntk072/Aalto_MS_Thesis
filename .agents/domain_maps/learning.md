# Learning Domain

`learning/` owns parameter fitting and learning-paradigm-specific model
adaptation. PPO/SAC is the production backend; a small Torch classifier is the
reference supervised backend for proving the shared fit and inference seams.

## Ownership

- PPO/SAC construction and policy customization.
- Gymnasium and vector-environment adapters used by RL rollouts.
- RL-specific rollout buffers and post-update hooks.
- Calls into reusable neural components owned by `models/`.
- A reference supervised Torch classifier and its self-describing checkpoint.

## Non-ownership

- Generic training workflow sequencing and campaign management.
- Trading, risk, execution, evaluation metrics or live broker semantics.
- Observation/action meaning; those schemas belong to `decisioning/`.

## Current source paths

- `learning/rl/agent.py` composes the current PPO/SAC implementations.
- `learning/rl/environment_adapter.py` provides the sharded SB3 VecEnv.
- `learning/rl/learner.py` implements `Learner.fit()`, persists the final SB3 checkpoint, and reports its content identity.
- `learning/rl/callbacks.py` owns SB3 rollout callbacks; `learning/rl/auxiliary.py` owns the auxiliary rollout trainer.
- `learning/rl/stopping.py` owns the SB3 episode-based early-stop rule.
- `learning/build.py` selects a paradigm-specific learner implementation.
- `learning/supervised/reference.py` implements a supervised Torch learner,
  provider, and checkpoint loader without Gymnasium, simulation, or SB3 imports.
- `learning/rl/ppo.py`, `hybrid.py`, and `rollout_buffer.py` own SB3-specific policy behavior.
- `models/encoders/` contains reusable Torch encoders behind a local
  feature-extractor contract; SB3/Gym imports and VAE extraction are under
  `learning/rl/`. `core.py` only preserves archived checkpoint import paths.

## Dependency direction

The RL backend may depend on `models/`, `decisioning/`, and simulation
environment contracts. Generic training workflows select a learner/backend.
`models/` and `decisioning/` must not depend on `learning/`.

## Tests and invariants

PPO/SAC and vector-environment tests remain under `tests/test_models/` and
`tests/test_train/`. `tests/test_learning_supervised_reference.py` proves that a
supervised artifact loads as a `DecisionProvider` and uses the common evaluation
rollout without an SB3 model or Gym adapter.

## Axis of change

- **Axis:** optimization paradigm and rollout implementation.
- **Owner:** `learning/` owns fitting; `models/` owns reusable architectures.
- **Stable boundary:** `Learner`/`FitRequest`/`FitResult` contracts plus
  `decisioning.DecisionProvider` for inference. Seed fitting uses
  `Learner.fit()`; checkpoint compatibility remains inside the SB3 backend.
- **Must not own:** trading semantics, generic evaluation, or campaign lifecycle.
## Change Map

| Task | Axis | Start file | Contract and focused test | Expected scope | Keep unchanged |
|---|---|---|---|---|---|
| Add or change a learner backend | fitting paradigm | `learning/build.py` | `Learner`, `FitRequest`, `FitResult`; `tests/test_train/test_fit_request.py` | one backend and fit contract tests | decision, trading, and simulation semantics |
| Add a supervised predictor | fitting paradigm and inference | `learning/supervised/` | `Learner` and `DecisionProvider`; `tests/test_learning_supervised_reference.py` | learner/provider and artifact tests | RL backend, trading semantics, and live integrations |
| Change PPO/SAC rollout behavior | RL implementation | `learning/rl/` | learning smoke and policy tests | RL backend modules and focused tests | generic evaluation and live adapters |

## Axis of Change

- **Axis:** fitting paradigm and backend-specific optimization behavior.
- **Owner:** `learning/` owns parameter fitting; `models/` owns reusable architectures.
- **Stable boundary:** `Learner`, `FitRequest`, and `FitResult`.
- **Expected edit radius:** one backend and its fitting contract tests.
- **Must not depend on:** trading semantics, campaign lifecycle, or evaluation internals.

## Verified Source Paths

- `learning/rl/agent.py`
- `learning/rl/checkpoint.py`
- `learning/rl/environment_adapter.py`
- `learning/rl/vae_extractor.py`
- `learning/contracts.py`
- `learning/build.py`
- `learning/rl/learner.py`
- `learning/rl/callbacks.py`
- `learning/rl/auxiliary.py`
- `learning/rl/stopping.py`
- `learning/supervised/reference.py`
