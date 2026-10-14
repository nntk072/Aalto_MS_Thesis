# RL Learning Quality Program: Implementation Status

This work is a separate research lineage. It does not change the historical
experiment registry or the readiness state of the thesis campaign.

## Implemented

- `config/quality_experiments.yaml` provides staged variants separate from
  `config/experiments.yaml`. `--quality-variant` applies the declared parent
  lineage. A later stage with a `*_winner` parent fails closed until a prior
  stage winner is explicitly selected in the registry.
- `account-state-v1` and `account-state-v2` have named, versioned coordinates.
  V2 exposes normalized account limits, realized daily loss, loss from initial,
  planned risk, active stop headroom, partial target state, remaining size,
  position age, targets, and management modes. Optional entry-state coordinates
  follow the account coordinates.
- New checkpoints and run manifests carry observation schema metadata. Resume,
  evaluation, and live inference validate it. An unversioned v1 checkpoint
  requires an explicit legacy opt-in; unversioned v2 is rejected.
- V2 state is wired through frame and shared-bundle environments. The default
  remains v1. V2 preserves the account's current realized-loss-only daily-loss
  definition; it does not change guardrails or FTMO evaluation semantics.
- `equity_delta_r` uses per-step equity change divided by fixed dollar risk and
  requires fixed-risk sizing. It rejects reward shaping combinations. Hard
  breaches still receive the existing terminal penalty, so the telescoping
  equality applies to non-breach, unshaped paths.
- Optional per-rollout diagnostics record action-coordinate distributions,
  saturation, rewards and reward components, advantages, entry funnel events,
  and constraint ratios/costs. They do not change policy updates.
- Per-step constraint telemetry reports normalized daily, maximum-loss, and
  trailing-drawdown use, positive increments, and hard-breach impulses. The
  environment's hard safety guardrails remain authoritative.
- Q4A semantic Gaussian heads are opt-in and preserve the declared Box action
  coordinate order and diagonal-Gaussian probability calculation.
- PPO rollout geometry and worker seeds are recorded in logs/manifests. The
  caller seed is now passed to SB3 so vector workers receive the reproducible
  seed sequence `seed + worker_index`.

## Not implemented or not established

- Q4B hybrid categorical-continuous PPO and Q5 PPO-Lagrangian are not
  implemented. Constraint telemetry is only the signal-collection foundation;
  there is no cost critic, cost rollout buffer, dual update, or learned shield.
- The registry's Q2/Q3/Q4/Q5/Q6/Q7 tournament variants have not been trained.
  Q2 rollout logging exists, but no winner has been selected.
- The new diagnostics have not been collected from an actual quality campaign.
- No new `srun`, Slurm allocation, or training campaign was launched. This
  implementation must not be interpreted as evidence that any policy learned,
  survived, traded profitably, or passed a thesis gate.
- The current project risk semantics remain unchanged. In particular,
  `AccountState.daily_loss` accumulates realized losing closes; floating loss is
  reflected in equity and drawdown but is not silently folded into that field.

## Verification

Focused CPU checks passed for v2 schema/vector contracts, no-position state,
short-side target distances, FSM coordinate placement, frame/bundle parity,
equity-delta reward accounting, registry inheritance and stage blocking,
manifest metadata, semantic-head gradients and save/load, PPO/vector-environment
wiring, action-schema persistence, and training smoke tests. These are
implementation checks, not model-quality results.
