# RL Learning Quality Program: Implementation Status

> **Plan routing:** The [master project execution plan](../thesis/master_project_execution_plan.md) is the single operational plan for Q0–Q7. This file records implementation status only; it does not authorize or document a launched research campaign.

> **Current-status pointer (2026-10-17 06:31 Helsinki):** This file retains
> dated snapshots that predate later Q5 implementation work. Use the current
> [Q0–Q7 implementation audit](../thesis/q0_q7_implementation_audit_2026-10-17.md)
> for the reconciled source status. Q5 primal-dual PPO is implemented with
> gamma-matched discounted cost updates, checkpointed gamma, fail-closed
> policy-only resume, and deterministic evaluation cost exports. The current
> dirty tree passed Ruff format/lint, Mypy, MI210 smoke, and 1,664 tests. Q5 has
> no research result and still awaits a frozen Q4 parent. The gate wrapper's
> final summary step refused to overwrite existing immutable artifacts; see
> the audit for exact hashes and limits.

This work is a separate research lineage. It does not change the historical
experiment registry or the readiness state of the thesis campaign.

## Q0–Q7 source audit — 2026-10-16

This audit compares `config/quality_experiments.yaml`, the training/model/env
implementations, and the focused tests. It is a source review only; it is not a
campaign run or runtime validation. User direction on 2026-10-16 makes every Q
stage compulsory for the overall mission. No Q-stage training result exists.

| Stage | Present in source | Remaining before a valid stage run |
|---|---|---|
| Q0 | A3/account-state-v1 control and diagnostics are registered. | Freeze source/data/config/seed/budget; run and package a full-split deterministic control. |
| Q1 | Versioned account-state-v2 coordinates, checkpoint checks, frame/bundle wiring, and focused tests exist. | Run the matched Q0/Q1 pair and review the behavioral/economic diagnostics; no quality campaign has tested it. |
| Q2 | `n_envs=16/32/64` arms are registered and the trainer records rollout geometry. | Arms are `implementation_unverified` and fail closed pending GPU-node config/trainer tests. Freeze optimizer-update interpretation (`n_envs × n_steps`, minibatch count, epochs), budget, and resource limits before comparison. Equal `n_envs` does not imply equal PPO update geometry. No runs or winner exist. |
| Q3 | RR and fixed-risk equity-delta reward implementations exist; registry arms are declared. | Arms are `implementation_unverified` and fail closed until GPU-node registry/trainer tests pass. Train from the selected Q2 parent and report economic metrics separately from reward. No runs or winner exist. |
| Q4A | Semantic Gaussian actor heads are wired and tests cover gradient, distribution, save/load, and schema behavior. | Arm is `implementation_unverified` until current GPU-node checks pass. Then train/evaluate from the selected Q3 parent. No run exists. |
| Q4B | Hybrid actor/distribution source and SB3 policy wiring are present, with a focused probability/gradient regression authored. | Arm is `implementation_unverified` and fail-closed. Verify decoder/category correspondence, PPO integration, deterministic evaluation, and checkpoint save/load on the GPU node before selecting it. |
| Q5 | Constraint-use telemetry and diagnostic logging exist. | **PPO-Lagrangian is not implemented.** No cost rollout/objective, dual update/state, checkpointed multiplier, or constrained-policy evaluation was found. The registry only adds `constrained_rl.enabled=true`, which the trainer does not consume; do not launch it. The status guard already blocks the placeholder; implement and validate the full algorithm before promoting it. |
| Q6 | Entry FSM, optional state observation, disabled-FSM control, and frame/bundle wiring exist. Registry includes visible, hidden, and no-FSM arms. | All three arms are `implementation_unverified` and fail closed pending GPU-node transition, observation, and config tests. Compare no-FSM vs hidden-FSM for eligibility and hidden-FSM vs visible-FSM for state visibility. No runs exist. |
| Q7 | LiteTCN and TCN are selectable; LiteTCN build/save-load and encoder tests exist. | Train/evaluate a matched comparison from the selected Q6 parent with the same data, action/reward/risk semantics, budget, and seed plan. No runs or winner exist. |

### Q5 implementation decision gate

The Q5 registry row is a placeholder, not a working algorithm. Current env
telemetry reports normalized daily/max-loss/trailing utilization changes and
hard-breach indicators; it is not consumed by PPO's objective. There is no
constraint cost rollout buffer, cost-value estimator, dual variable/update,
Lagrangian policy objective, checkpointed multiplier state, or evaluation
reporting for learned costs. The experiment specification must define cost
signals, episode aggregation and budgets, discounting, dual initialization,
step size/bounds, and combination of simultaneous account constraints. These
choices define the method; implement and test them as one integrated Q5 path.
The registry's `planned` status and selection guard prevent
`constrained_rl.enabled` from being silently added to ordinary PPO. Q5 is not
selectable.

Stage-parent `*_winner` references intentionally fail closed until the registry
is changed to the exact selected parent. Preserve that rule and record every
winner/no-winner decision before opening a child stage. The registry now also
rejects variants whose status is not an explicitly selectable implementation
state (`control`, `implemented_unit_tested`, or
`existing_encoder_implementation`). New `implementation_unverified` arms stay
blocked until their GPU-node checks pass and the status is promoted. This guard
and the new registry regressions have not passed on an authorized GPU node.

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

- Q2/Q3/Q4/Q6 arms and Q4B source are not selectable until their registry and
  integration regressions pass on a GPU node. Q5 PPO-Lagrangian is not
  implemented. Constraint telemetry is only the signal-collection foundation;
  there is no cost critic, cost rollout buffer, dual update, or constrained
  policy objective.
- No Q0–Q7 variant has been trained as a quality campaign. Q2 rollout logging
  exists, but no winner has been selected. Existing thesis campaign workers do
  not count as Q-program runs.
- Q6's visible, hidden-state, and no-FSM arms are registered; their regression
  and complete three-arm config have not passed validation on a GPU node. The
  comparison still has no run or result.
- The new diagnostics have not been collected from an actual quality campaign.
- No real `srun` allocation or training campaign was launched. `srun --test-only`
  was used only to query scheduler forecasts. This
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

## Implementation update — 2026-10-17 00:00 Helsinki

- The registry separates source presence from validated availability. Q2, Q3,
  Q4A/Q4B, and all three Q6 arms use `implementation_unverified`; the registry
  rejects those variants until their focused GPU-node checks pass. Q5 remains
  `planned` and also fails closed. Q0 and Q7 remain selectable source controls,
  not completed research stages.
- Q6 has the required three-arm design: no FSM, FSM hidden from the policy, and
  FSM visible to the policy. The config regression verifies the arm settings.
- New regressions cover Q2 geometry mapping, the Q6 three-arm config, blocking
  unverified variants, and Q5 placeholder rejection. They are authored but
  have not been run on a GPU node.
- Q4B hybrid categorical-continuous policy code exists but remains unverified.
  Q5 still needs an integrated constrained PPO path: cost collection, cost
  return/value estimation, dual update, constrained objective, multiplier
  checkpointing, and evaluation reporting. Telemetry is not an implemented
  Lagrangian method.
- `git diff --check` passed. No Python tests or training were run in this
  snapshot.

## Follow-up source audit — 2026-10-17 00:07 Helsinki

This is a source/config/test inventory, not runtime validation. The current
login check found no active GPU allocation to reuse; job `21004765` remains
pending because its requested node is down/drained/rebooting or reserved. No
new allocation or test step was started.

- Q0 is a registered A3/account-state-v1 source control; it is not a completed
  frozen run. Q1's account-state-v2 implementation and metadata checks exist,
  but no matched Q0/Q1 campaign result exists.
- Q2's 16/32/64 arms set `env.n_envs`, and training logs its effective
  `n_envs`, per-env `n_steps`, aggregate rollout, minibatches, and updates.
  The arms remain fail-closed; they need current integration checks and a
  frozen equal-budget/update interpretation before a comparison.
- Q3's reward/risk implementations exist in the environment. Its registered
  arms remain fail-closed and have no selected Q2 parent or experiment result.
- Q4A semantic Gaussian heads and Q4B hybrid categorical/continuous heads
  have source and focused test files. They are not currently selectable. Q4B
  still needs current full policy integration, action-decoder/category
  correspondence, PPO update, deterministic prediction, and checkpoint checks.
- Q5 Lagrangian PPO is **not implemented**. Environment constraint telemetry
  is present, but there is no cost-aware PPO objective, dual update, persisted
  multiplier, or constrained evaluation. Its cost budgets and episode
  aggregation are method choices that must be specified before training; the
  registry correctly blocks the placeholder.
- Q6's FSM and visible/hidden/no-FSM configs exist, but all arms remain
  fail-closed and need transition, observation, frame/bundle parity, and
  trainer integration checks.
- Q7's LiteTCN/TCN implementations exist; this does not establish a matched
  comparison or winner. Parent selection still depends on completing Q6.

No Q stage is complete as a research stage: there is no frozen Q0 run, no
stage winner, and no Q0–Q7 campaign evidence. This pass made no Q algorithm
changes because no Q5 cost-budget/method specification or selected parent
lineage is frozen, and the required GPU-node validation environment is not
currently available. The one-minute Slurm allocation rule was tightened in
`AGENTS.md` and `.agents/rules/triton-slurm.md`; it now explicitly requires
`--immediate=60` for supported blocking requests, fresh per-node CPU/RAM/GPU
inventory before each alternative, and an A100-or-better/MI210 fallback.

## Follow-up source, defect, and GPU-node audit — 2026-10-17 01:42 Helsinki

The one-minute rule was verified in both `AGENTS.md` and
`.agents/rules/triton-slurm.md`. It limits only allocation wait/polling, requires
fresh per-node CPU/RAM/GPU inventory before each alternative, and does not
authorize cancellation. An approved MI210 allocation (`21007645`, `gpuamd1`,
8 CPUs, 128 GiB requested RAM, one MI210 for six hours) was used. The node has
16 CPUs/250 GiB total, ROCm 6.3.1, and an MI210 (`gfx90a`, 64 GiB VRAM).

The repository's current lock selects CUDA 13.0, so it cannot be used on MI210.
Added `scripts/setup/setup_venv_mi210.sh`, which creates a separate Python 3.12
test environment from the locked non-accelerator dependencies and installs
PyTorch 2.7.1+rocm6.3. A real 256×256 matrix multiply succeeded. The source and
test suite below ran in that environment, but this is not exact locked-stack
validation or an MI210 throughput/trainer qualification.

| Stage | Source and checks now confirmed | Still open before a research result |
|---|---|---|
| Q0 | Registered A3/account-state-v1 source control. | Freeze data/source/config/seed/budget, run full chronological training, deterministic full-split evaluation, and package provenance/diagnostics. No Q0 run exists. |
| Q1 | account-state-v2 schema, checkpoint metadata, and frame/bundle wiring. Focused schema/parity tests passed. | No matched Q0/Q1 policy run or selected disposition. |
| Q2 | 16/32/64 arms and effective PPO geometry logger exist. Added tests for logged aggregate rollout/minibatch/update counts and mismatch rejection. | Arms remain fail-closed. Freeze the Q1 parent, equal-budget/update interpretation, resource limits, and learning-quality review. No geometry run or winner exists. |
| Q3 | RR and fixed-risk equity-delta reward code and registered arms exist; equity-delta accounting tests passed. | Arms remain fail-closed. Freeze the Q2 winner and predeclare the multi-arm family, budgets, and separate reward/economic reporting. No Q3 run exists. |
| Q4A | Semantic Gaussian actor code; PPO update, distribution, gradients, save/load tests passed on CPU and MI210. | Still fail-closed pending selected Q3 parent and full deterministic trading-environment evaluation. No policy result exists. |
| Q4B | Hybrid categorical/Gaussian code; PPO update, gradients, save/load tests passed on CPU and MI210. Added checks that TP categories map through the existing TP decoder. | Still fail-closed pending selected Q3 parent, complete real-environment action semantics/deterministic evaluation, and stage acceptance. No policy result exists. |
| Q5 | Registry placeholder is correctly blocked. Constraint-use telemetry exists. | **PPO-Lagrangian is not implemented.** There is no constraint-cost rollout, cost critic/return, dual update, multiplier checkpoint state, or constrained evaluation. Define and review cost signals, episodic aggregation, budgets, discounting, initialization/rate/bounds, and multiple-account-constraint combination before implementing it. |
| Q6 | FSM lifecycle, visible/hidden observation wiring, and no-FSM control are present. Added frame/bundle parity checks through FLAT→CANDIDATE→ARMED→IN_POSITION for hidden and visible state. | Three registered arms remain fail-closed. Freeze the Q5 parent and complete trainer/config/reporting review; no Q6 comparison exists. |
| Q7 | LiteTCN/TCN encoders and their focused encoder tests passed. | No frozen Q6 parent, matched policy comparison, run, or winner exists. |

The audit also found and fixed three source/test defects: `allow_ema_exit=false`
was not forwarded to either the trainer's `TradingEnv` or deterministic eval
kwargs; the variant resolver checked an arm's status before reporting its
unresolved `*_winner` parent; and two Q6 registry tests treated flat dotted
override keys as nested YAML. The Q4B target-decoder regression and Q2/Q6
coverage gaps described above were added.

Validation command on the MI210 node:

```bash
PYTHONPATH="$PWD" /scratch/work/nguyenl37/.venv-mi210/bin/python -m pytest \
  tests/test_envs/test_eligibility_wiring.py \
  tests/test_train/test_quality_registry.py \
  tests/test_envs/test_observation_schema.py \
  tests/test_envs/test_bundle_env_equivalence.py \
  tests/test_envs/test_equity_delta_reward.py \
  tests/test_models/test_semantic_actor.py \
  tests/test_models/test_hybrid_actor.py \
  tests/test_models/test_encoders.py \
  tests/test_models/test_lite_tcn_agent.py \
  tests/test_envs/test_entry_state_machine.py \
  tests/test_rebuild_named_prebuilt_feature_caches.py \
  tests/test_train/test_train_rl_decomposition.py \
  tests/test_features/test_session_ohlc.py \
  tests/test_features/test_feature_cache_and_fit_scope.py \
  tests/test_features/test_cache_hash_full_column.py -q
```

Result: **139 passed**, 8 PyTorch Transformer nested-tensor warnings, in 18.22
seconds. Semantic-head and hybrid policy suites included GPU PPO updates and
checkpoint round trips on MI210. These are bounded implementation checks only.
No Q stage has been trained; there is no Q winner, economic evidence, or
Lagrangian result. Do not start Q-stage training before the frozen stage parent
and the master plan's resource/entry gates are met.

## Current-source causal audit and feature-loop repair — 2026-10-17 02:21 Helsinki

The MI210 validation allocation was also used to complete the P1 real-market
checks. To make full-history checks tractable, the FVG and IFVG builders now
write their existing ordered state-machine results into local NumPy arrays
rather than repeatedly resolving pandas scalar `.iat`/`.at` access. This
preserves the row/zone iteration order and formulas; it is an implementation
change, not a policy or feature-definition change. No controlled before/after
throughput benchmark was collected, so no speedup is claimed.

| Evidence | Result | Limit |
|---|---|---|
| `tests/test_features/test_variants.py`, `test_assembled_causality.py`, `test_po3_state.py`, `test_po3_ifvg_state.py` | 42 passed on MI210, 3.47 s. | Focused FVG/IFVG and assembled-causality contracts only. |
| `docs/thesis/g1_feature_prefix_post_repair_2026-10-17.json` | 12/12 cutoffs pass over four feature profiles representing all 18 resolved variants; 528,760 M1 rows, feature version `v22-causal-configured-ny-session`; source CSV hashes are recorded in the report. | Sampled prefix invariance, not exhaustive proof across all operations or every row. |
| `docs/thesis/g1_real_market_fold_fit_scope_post_array_opt_2026-10-17.json` | 4/4 profiles pass one early purged fold: 41,920 training rows and 6,000 validation rows each. Poisoning validation OHLC leaves training features bit-identical and changes validation features. | One fold per profile; not five-fold evaluation, strategy performance, or OOS confirmation. |
| `docs/thesis/data_quality_audit_2026-10-16_v4.json` | Both M1 files have zero parse failures, invalid OHLC ordering, negative tick volume, or negative spread; `VOL=0` is recorded and interpreted as absent real volume for this CFD export. | PARTIAL: no provider/license record, authoritative calendar, broker DST proof, or tick-file rescan. |
| `docs/thesis/prebuilt_cache_rebuild_2026-10-17_mi210.json` | Transactional rebuild passed for three caches, each with 528,760 rows: baseline 194 columns, Idea 1 532, Idea 2 538. Hashes, parquet round trips, raw inputs, and loaded-source provenance are recorded. Old caches/sidecars were backed up under `data/prebuilt_feature_cache/backups/pre-repair_2026-10-17_mi210_21007645/`. | Built on ROCm MI210 with the locked CPU dependencies and PyTorch 2.7.1+rocm6.3; not an exact CUDA-lock or trainer qualification. |

The M1 audit initially failed because it compared equivalent time strings
`08:30` and `08:30:00` literally. It now normalizes configured times to
seconds and emits a new immutable v3 report; v2 is preserved. Formatting and
lint pass for the touched Python files, and `git diff --check` passes.

After rebuilding the caches, the combined Q/feature regression set passed:
**181 passed, 8 expected Transformer nested-tensor warnings, 17.50 seconds** on
job `21007645`. This is a focused source gate, not the full repository CI suite.

The required full-tree commit checks also passed on the MI210 node: Ruff format
and lint passed across 448 files, and staged full-tree Mypy 2.3.0 reported
`Success: no issues found in 457 source files`. The staging directory was
node-local and temporary. The repository-wide `pytest tests/ -v` CI suite has
not been run in this agent pass.

Q0–Q7 remain **not complete** as empirical stages. Q5 PPO-Lagrangian remains
unimplemented: the source has normalized constraint telemetry but no cost-aware
rollout/critic, Lagrange update, persisted multiplier, or constrained
evaluation. It stays fail-closed until cost budgets, aggregation horizon,
discounting, dual update/bounds, and the frozen Q4 parent are resolved under the
master plan. The new causal evidence permits continued P1 preparation; it does
not make the contaminated 18-run screen admissible or authorize a new campaign.
