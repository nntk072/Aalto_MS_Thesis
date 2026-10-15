# Aalto Thesis Trading System — Master Execution Plan

**Status:** ACTIVE — thesis/campaign readiness remains **NOT READY**
**Revision:** 2026-10-15
**Purpose:** One ordered plan for correctness, campaign evidence, throughput work, and the Q0–Q7 research program.

## 1. Authority and how to use this file

This is the sole operational roadmap. Use it for current priority, dependency
order, stage status, and next action. Do not run separate work queues from the
older plan documents in parallel.

The detailed plans below remain technical references and historical evidence;
they are not independent authorization to launch work. The
[implementation ledger](implementation_ledger_2026-10-13.md) remains the
append-only evidence record for code, tests, artifacts, and dated observations.
The [independent review checklist](independent_research_review_checklist_2026-10-14.md)
remains the reviewer-facing checklist. Keep readiness **NOT READY** until this
plan's final acceptance conditions are supported by evidence.

### Consolidated source documents

| Previous document | Disposition in this plan |
|---|---|
| `/scratch/work/nguyenl37/Agent_Implementation_Plan.md` | Core correctness and validity requirements; WP00–WP13 implementation status is summarized here. Its G0–G4 campaign gates are retained below. |
| [`18_variant_training_readiness_plan.md`](../training/18_variant_training_readiness_plan.md) | Current campaign contract, failed-pilot history, readiness evidence, and worker/evaluation requirements. |
| [`18way_runtime_scaling_screen_2026-10-14.md`](../training/18way_runtime_scaling_screen_2026-10-14.md) | Historical 100k runtime projection and its limitations; superseded by current 20M live progress for the current-run estimate. |
| [`fast_training_branch_benchmark_20261015.md`](../training/fast_training_branch_benchmark_20261015.md) | GH200 PPO-core results for vector backend and concurrency; incorporated below with scope limits. |
| [`trainer_speed_optimization_plan.md`](../training/trainer_speed_optimization_plan.md) | Campaign-speed measurement sequence after the active campaign and continuation supervisor finish. |
| [`gh200_pre_q0_throughput_plan.md`](../training/gh200_pre_q0_throughput_plan.md) | GH200 experiments already measured, their limits, and unresolved work before Q0. |
| [`triton_evaluation_acceleration_agent_plan.md`](../training/triton_evaluation_acceleration_agent_plan.md) | Report/evaluation profiling and numerical-parity work not yet accepted. |
| [`triton_gpu_training_agent_plan.md`](../training/triton_gpu_training_agent_plan.md) | Controller durability and campaign audit requirements. Its historical 18 × 5 production matrix is superseded by the current one-seed screening decision; do not launch 90 runs from that document. |
| [`triton_ram_optimization_plan.md`](../training/triton_ram_optimization_plan.md) | Shared-bundle parity and cgroup memory acceptance requirements. |
| [`sharded_bundle_vecenv.md`](../training/sharded_bundle_vecenv.md) | Opt-in worker-sharding contract, semantics, and acceptance gate. |
| [`prebuilt_feature_caches.md`](../training/prebuilt_feature_caches.md) | Named baseline/Idea 1/Idea 2 cache locations, content-hash checks, and fail-closed behavior. |
| [`rl_learning_quality_implementation.md`](../training/rl_learning_quality_implementation.md) | Implemented versus not-yet-implemented Q-program foundations; no model-quality results. |
| [`quality_experiments.yaml`](../../config/quality_experiments.yaml) | Defines Q0–Q7 experiment variants and parent controls. It is a registry, not a launch instruction or a result. |

This roadmap now contains the operative requirements and acceptance checks
from every plan listed above. The detailed files remain historical source
records and evidence pointers; you do not need to read them to know the current
plan. Do not delete those records. If a historical instruction conflicts with
this roadmap or a later author decision recorded in the ledger, this roadmap
and the later author decision control.

## 2. Frozen research and operating decisions

- Each policy must traverse the complete chronological training split before
  its deterministic in-sample and report-only test evaluations.
- The current screening campaign is 18 variants, seed 42, 20M PPO timesteps
  each, `n_envs=64`, shared-bundle backend, and full-split episode mode.
- This one-seed campaign screens variants. Run a five-seed robustness study
  only after at least one candidate completes training and both evaluations
  successfully; do not expand the current matrix to 18 × 5 by default.
- Positive net PnL and trade count are reported metrics, not hard pass gates.
  There is no 50-trade minimum. Apply the agreed survival/equity gates without
  silently changing them.
- The currently available calendar ends 2026-07-03. No post-July data is a
  prerequisite. The already inspected 2026 test interval is
  selection-acknowledged, exploratory data; it cannot be called a pristine
  confirmatory holdout. Do not relabel it or select a model from it.
- The pure legacy `ladder_a0_flat` is the reference baseline for the current
  screening campaign. A Q0 control is separately specified in the quality
  registry and must not be assumed equivalent to A0.
- Preserve Idea 2's after-IFVG distribution entry semantics and the existing
  causal, multi-timeframe feature contract.
- Reuse existing allocations and supervisors. Do not submit or cancel Slurm
  jobs without the explicit authorization required by `AGENTS.md`. Do not
  change code/config files included in a live campaign fingerprint.
- Training monitoring is on demand, not continuous. At each requested check,
  reconcile Slurm, the supervisor, campaign manifest, worker logs, checkpoint
  lineage, and fresh artifact timestamps; the campaign status JSON may be stale.

## 3. Current verified state

### Repository and active campaign — checked 2026-10-15 20:48 UTC

- Checkout: `/scratch/work/nguyenl37/Aalto_MS_Thesis`, branch
  `codex/fast-training-wp0`, HEAD `061987e15b124a616e924d014a0be778826ca305`.
- Existing user/work-in-progress edits are preserved in
  `docs/training/gh200_pre_q0_throughput_plan.md`,
  `scripts/bench/measure_ppo_throughput.py`, and
  `tests/test_bench/test_measure_ppo_throughput.py`.
- Slurm job `20974801` is RUNNING on `gpuarm1` and `gpuarm2`: one GH200, 48
  CPUs, and 512 GiB RAM per node; six-hour allocation. At this check it had
  run 3h23m. All 18 worker logs were fresh.
- Latest worker logs showed approximately 3.83M–6.88M of 20M steps per
  variant, with reported worker FPS around 321–575 (median about 498). This is
  interim progress, not completed training or policy evidence. The on-disk
  `campaign_status.json` timestamp was stale relative to live worker logs.
- The first A0 full-split episode record contains 64/64 environments, each
  traversing 99,504 training bars and ending at `data_end`; the record also
  logs trailing-drawdown breaches. This demonstrates chronological training
  exposure for that A0 episode, not survival, profitability, or a passing
  final evaluation. Other variants had not yet produced a recorded complete
  split at the check.
- No final train/test reports were present. Repeated dashboard warnings say
  `evaluate_model()` rejects `observation_schema_version`. Static inspection
  confirms `_build_eval_common()` also passes the unsupported
  `observation_age_scale_bars`; final deterministic train/test calls use this
  same kwargs mapping and are expected to fail unless corrected. A correction
  exists in the separate `/scratch/work/nguyenl37/Aalto_MS_Thesis-evalfix`
  worktree (`codex/eval-schema-fix`); focused `tests/test_rollout.py` passed
  3/3 and Ruff passed there. It is not applied to the active checkout. Preserve
  the live campaign fingerprint, then use its saved checkpoints for a
  provenance-linked evaluation recovery if final calls fail.
- The full-split continuation supervisor is alive in tmux session
  `fullsplit-supervisor`; its script waits for the active allocation and
  resumes the same campaign across allocations. Do not start a duplicate
  supervisor or second run.

### Core implementation status

WP00–WP13 are marked **PASS** in the ledger as bounded implementation
packages. This is code and focused-verification status, not completion of the
end-to-end thesis plan. WP00 lacks a new immutable artifact freeze; 42 legacy
archives lack configs; WP10 has no provenance-complete treatment/control
policy pair; WP11 has no certified historical `run_manifest.json`; WP12 is
bounded deterministic parity, not exhaustive production/live parity.

### Throughput evidence already available

- Device-resident PPO rollout staging is enabled by default on the active
  branch. Two paired GH200 PPO-core runs measured a 59.1% mean throughput gain
  over the CPU rollout buffer, with exact minibatch parity and about 0.71 GiB
  more peak VRAM. This is not campaign-level or policy-quality evidence.
- Larger batch/rollout geometries have short PPO-core measurements, but no
  candidate has been accepted on learning quality.
- Pinned prefetch was not justified by measured copy time; deferred scalar
  synchronization showed no repeatable gain; broader precision showed no
  measured gain; GPU-vectorized environment work was not justified by phase
  timing. `torch.compile` was blocked by the missing node C compiler, CUDA
  Graphs were not run, and JAX/XLA remains deferred.
- The current 18-worker campaign is not a matched throughput comparison to
  the earlier three-worker run. The current exact-config benchmark and
  end-to-end quality review remain open.

## 4. One ordered work plan

Only the first active stage is a work queue. Later stages remain gated until
their entry conditions are met.

### Stage A — Preserve and finish the active full-split screen

**Status: IN PROGRESS.**

1. Leave job `20974801`, its workers, tmux supervisor, outputs, source files,
   and campaign fingerprint untouched. Use the existing supervisor's
   allocation-resume path; do not submit another job manually.
2. At an on-demand status check, record per-variant timesteps/FPS, completed
   chronological episode coverage, worker exit state, checkpoint hashes,
   allocation status, and resource samples. Do not trust a stale summary over
   current logs.
3. Allow all 18 variants to reach their requested budget or record a
   terminal, evidence-backed failure. Preserve failed logs and any valid
   checkpoints.
4. Do not interpret the dashboard-prefix errors as final evaluation results.
   After the active source boundary is safe, correct the evaluation argument
   contract and verify that observation schema and age scaling match training.
5. Produce deterministic train-in-sample and report-only test evaluations for
   every valid final checkpoint. If the controller marked a worker failed at
   final evaluation, recover from its saved checkpoint without retraining if
   provenance and deterministic evaluation can be verified. Clearly separate
   training completion, evaluator recovery, and controller status.
6. Audit every report against the effective run config, data and feature/cache
   hashes, checkpoint lineage, split boundaries, costs, action schema, and
   raw trade/equity reconstruction. The exposed 2026 test remains exploratory.

**Exit evidence:** 18/18 terminal statuses reconciled; each successful run has
a valid loadable checkpoint, complete train-split exposure, deterministic
train and test reports, and an auditable failure reason where unsuccessful.
Passing this screen does not itself make a variant confirmatory or the thesis
READY.

### Stage B — Close the original plan's release gates G0–G3

**Status: OPEN; WP00–WP13 implementation rows are already marked PASS.**

| Gate | Remaining acceptance work |
|---|---|
| G0 — Correctness | Resolve the live evaluation-kwargs defect; run relevant regression tests; verify position/fill/reward reconciliation and causal execution contracts in the actual report path. |
| G1 — Validity | Reconcile full-split training, fold-specific feature fitting, strict eligibility, canonical metrics, and actual train/test reports. Preserve exploratory-vs-confirmatory boundaries. |
| G2 — Design | Freeze one matched control/treatment pair with exactly one declared experimental dimension, seed, budget, costs, checkpoint rule, and selection rule. The current 18-rung screen is not automatically a matched causal comparison. |
| G3 — Reproducibility | Verify run/attempt manifests, code/config/data/cache identities, loadable checkpoint hashes, raw metric reconstruction, and trial/holdout ledger for the completed screen and any follow-up. Do not backfill provenance that was never recorded. |

### Stage C — Research review and confirmatory-data decision

**Status: OPEN.**

1. Assemble the completed campaign reports, source/config boundary, failed
   pilots, execution/reward/termination limitations, and metric definitions
   into the existing independent-review checklist.
2. Obtain independent research review before candidate claims or thesis
   conclusions.
3. Establish whether an actually untouched holdout exists within authorized
   data. If not, continue to label all current evaluations exploratory and
   obtain a genuinely unseen interval before making confirmatory claims. No
   requirement for post-2026-07-03 data is imposed by this roadmap.

**Exit evidence:** reviewer disposition recorded; no historical OOS period is
recast as pristine after it has been inspected.

### Stage D — Candidate robustness and campaign gate G4

**Entry:** at least one screening candidate completes the full train/test
process and passes the predeclared survival/equity gates.

1. Use a declared baseline and treatment with paired seeds and identical
   dataset, split, training exposure, costs, PPO budget, checkpoint rule, and
   evaluation procedure.
2. Run the planned five-seed robustness study only for the candidate/control
   comparison authorized after screening. Add cost stress and report resource
   use over a complete training/evaluation cycle.
3. Select settings using training/validation only. Do not rank candidates by
   the exposed report-only test period.

**Exit evidence:** G4 campaign artifacts, paired-seed statistics, declared
controls, cost sensitivity, provenance, and independent review. A negative
result is valid evidence and does not become a success by extending training.

### Stage E — Finish pre-Q0 performance and resource plans

**Entry:** Stage A is terminal and its checkpoints/reports are preserved.

1. Complete the exact-config phase profile and benchmark cleanup in
   `trainer_speed_optimization_plan.md`. Measure environment/reset, rollout,
   PPO update, callbacks/evaluation, checkpointing, and report-export time on
   the same effective source/config/data/device.
2. Re-measure concurrency and `n_envs` at fixed semantics. Compare aggregate
   steps/s and end-to-end time, per-worker rates, CPU pressure, GPU/VRAM,
   cgroup RAM, and OOM counters. Use existing/authorized allocations only.
3. Finish shared-bundle acceptance only with all relevant variant observation,
   reward, execution, termination, train/test, and checkpoint parity; do not
   promote on RAM savings alone.
4. Finish evaluation acceleration with numerical parity, matched saved-model
   timings, artifact inventory, resource measurements, and resumability. Keep
   plot omission/deferment distinct from numerical evaluation.
5. Close remaining GH200 pre-Q0 items according to the measured plan. Keep
   `torch.compile`, CUDA Graphs, JAX/XLA, precision changes, or GPU environment
   rewrites deferred unless a measured bottleneck and adequate toolchain justify
   them. No optimization is accepted solely for higher FPS.

**Exit evidence:** reproducible end-to-end time/resource comparisons and no
policy-quality, semantic, evaluation, or provenance regression. This stage
does not launch Q0 or change the live screen.

### Stage F — Run the Q0–Q7 learning-quality program sequentially

**Status: NOT STARTED as a research campaign.** The registry's
`implemented_unit_tested` labels mean code availability only, not an evaluated
model result. Q0 requires the author's direction after the prior stages; Q1
and later stages do not run ahead of their registered parent/control.

| Stage | Registered question | Advancement rule |
|---|---|---|
| Q0 | A3-based `q0_control_a3` reference with account schema v1 | Freeze config, seed/budget, data identity, and checkpoint/evaluation rule. Establish a valid control result before comparing Q1. |
| Q1 | `q1_state_v2` observation/account-state schema | Change only the registered schema dimension versus Q0; require parity and quality review. |
| Q2 | `n_envs` 16 / 32 / 64 | Compare controlled rollout geometries; choose only after throughput and learning-quality review. |
| Q3 | Reward/risk alternatives | Compare each reward/risk setting to the declared Q2 winner; do not combine unregistered changes. |
| Q4 | Semantic actor heads / hybrid distribution | Use the Q3 winner as control; verify action semantics and numerical stability. |
| Q5 | Lagrangian constraints | Compare against the Q4 winner with the constraint protocol frozen. |
| Q6 | Visible entry state machine / no-FSM control | Change entry-state handling only; preserve after-IFVG behavior unless separately approved. |
| Q7 | LiteTCN versus TCN encoder | Compare using the Q6 winner and equal data, budget, seeds, and evaluation. |

At each stage, keep a single-factor treatment/control, one declared discovery
seed for initial screening, and a five-seed follow-up only after a valid
full-split train/test result. Use locked or exposed OOS strictly according to
the validation protocol; never use it to select the experiment.

## 5. Global acceptance and stop conditions

The thesis system is READY only when all are true:

1. G0–G4 have evidence at their required scope, not merely passing unit tests.
2. The candidate has full chronological training exposure and deterministic
   train/test evaluation with reconciled, provenance-complete artifacts.
3. Survival/equity gates pass under the frozen rules; PnL and trade count remain
   reported metrics unless the author changes the protocol before a new run.
4. Matched controls, paired-seed robustness, cost stress, and the required
   independent review are complete.
5. Holdout status is honest. The previously inspected 2026 interval is never
   described as pristine confirmation.
6. Required parity, resource sizing, report-generation, and reproducibility
   gates for the selected implementation are complete.

If a gate fails, retain the failure, diagnose from evidence, and define the next
registered experiment. Do not weaken the gate, repeat a failed run with hidden
settings, or turn an unfinished allocation into a success claim.

## 6. Single-source status update procedure

When asked for an update, refresh Slurm and the existing tmux supervisor,
reconcile all 18 worker logs and output artifacts, then update this plan's
current stage and next action. Append detailed measurements and hashes to the
implementation ledger. Keep the old plans as reference material and do not
create another parallel plan. Do not update campaign source/config while the
current fingerprinted campaign or its continuation supervisor is active.

### Current next action

Preserve job `20974801` and the existing `fullsplit-supervisor`. On the next
requested status check, refresh worker progress and resource samples. Work on
the final-evaluation compatibility correction only in the separate worktree
until the live campaign reaches a safe source boundary. Do not start Q0–Q7.

## 7. Planning estimate and Goal activation

**Estimate refreshed:** 2026-10-15 20:48 UTC. These are planning ranges, not
promised completion dates. Keep measured runtime, scheduler wait, research
review, and conditional experiments separate.

### Current campaign and full roadmap horizon

At the latest live check, workers ranged from 3.83M to 6.88M of 20M steps.
Dividing each worker's remaining budget by its latest logged FPS gives an
estimated **6.3–14.0 hours of additional active training**, with the slowest
worker setting the campaign finish. This assumes the latest FPS holds; it is a
projection, not a measured finish time. The current six-hour allocation had
about 2h37m left, so the slowest worker needs roughly two more six-hour
allocation windows if they are reacquired promptly. Queue gaps can stretch that
to several calendar days. Final train/test evaluation, the known evaluator
argument defect, recovery, and artifact audit add work after the PPO steps end.
Allow **about 3–7 calendar days for Stage A**, assuming the existing supervisor
continues to reacquire suitable nodes.

| Work after / alongside Stage A | Planning range | Main uncertainty |
|---|---:|---|
| Close G0–G3: evaluator correction, regression checks, report/lineage audit, matched design | 1–2 weeks | Quality and provenance defects may require recovery or another run. |
| Independent research review and holdout disposition | 1–4+ weeks | Reviewer availability; whether an untouched authorized interval exists. If none exists, confirmatory readiness has no finite estimate until an acceptable holdout is available. |
| G4: five-seed baseline/treatment robustness plus cost stress | 3–7 days | Ten 20M runs are the planning case; concurrent throughput and allocation gaps are not yet measured for this exact matrix. |
| Pre-Q0 throughput, shared-bundle parity, and evaluation acceptance | 3–7 working days | Some PPO-core work is complete, but end-to-end parity and exact-campaign timing remain open. |
| Q0–Q7, sequentially | 2–5 weeks | The registry has 15 listed variant/control entries across seven gates; each later stage depends on the prior winner. The estimate assumes one 20M discovery run per listed entry where needed, reuse of valid parent controls, no broad redesign, and prompt stage decisions. |

With those assumptions, budget **about 8–12 weeks of calendar time** to work
through the full roadmap and produce a defensible final readiness verdict.
Four to six weeks is an optimistic case requiring prompt allocations and
review, no failed candidate gates, an acceptable holdout within authorized
data, and few recovery runs. More than three months is plausible if a candidate
fails survival/equity gates, data/provenance defects require retraining, or the
research review/holdout is delayed. If no model passes the required gates, the
work can still finish with a well-supported **NOT READY** conclusion; there is
no honest finite estimate for achieving **READY** before those results exist.

This is elapsed project time, not continuous Codex execution. Slurm jobs run
independently. Codex Goals persist across turns and may continue at safe idle
boundaries while active, but this project requires on-demand monitoring and
will pause at user/reviewer/data/resource gates.

### Goal to activate after the user clears prerequisites

Keep the existing Codex Goal paused until the user asks to start. Use the
following as the replacement Goal wording when activation is appropriate:

> Execute `/scratch/work/nguyenl37/Aalto_MS_Thesis/docs/thesis/master_project_execution_plan.md`
> in dependency order. Verify each stage against repository artifacts, tests,
> campaign logs, checkpoints, and review evidence. Preserve existing jobs,
> user data, and working-tree changes. Monitor the campaign only when asked;
> never submit or cancel Slurm work without explicit user authorization. Do not
> start Q0–Q7 until all prerequisite gates pass and the user authorizes the
> research program. Distinguish measurements, projections, exploratory results,
> and confirmatory evidence. Mark READY only if every stated acceptance gate
> passes; otherwise finish with the evidence-backed NOT READY verdict and the
> exact external input needed to proceed.

The current Codex Goal is already **PAUSED**. The available Goal controls can
read its objective and change its status, but cannot edit its objective in
place. Therefore this draft updates the durable roadmap without resuming or
falsely completing the existing Goal; the replacement wording can be applied
when the user is ready to start it.

## 8. Self-contained technical requirements from the consolidated plans

This section is the working reference for implementation, experiment design,
training infrastructure, reports, and Q0–Q7. The linked former plans are not
needed to interpret these requirements.

### 8.1 Original implementation plan: WP00–WP13 status

The ledger marks WP00–WP13 **PASS as bounded implementation packages**. That
means the scoped code contracts and focused checks recorded there passed; it
does not mean the full thesis system, all historical artifacts, or live trading
parity is certified. Do not repeat a passed package unless new evidence finds
a defect.

| Package | Completed bounded contract | Remaining limit that matters to readiness |
|---|---|---|
| WP00 — baseline freeze | Historical baseline/source review and reproduction evidence recorded. | No new immutable artifact freeze; some source/data/checkpoint identities were unavailable. |
| WP01 — action schema | Named schema/layout metadata, explicit legacy migration, checkpoint validation, archive inventory. | 42 historical archives have no run config; they remain unevaluable. Rerun the current pure-A0 action/feature audit if the action implementation changes. |
| WP02 — fill accounting | Stable position/fill IDs, quantity-aware partial realization, aggregate reward PnL, completion-only cooldown. | Exhaustive engine parity is separate; all final reports still need raw ledger reconstruction. |
| WP03 — trailing stop | Causal confirmed swing references, long/short direction, next-bar activation, no loosening, initial risk preserved. | Bounded tests do not prove every live/evaluation path. |
| WP04 — target execution | Slot-aware tick/bar fills, chronological crossed targets, residual stop close, disabled/hit-slot preservation, conservative stop-first OHLC ambiguity. | The simplified benchmark backtest remains single-target; do not equate it with the environment's multi-slot execution. |
| WP05 — clocks and costs | Causal fill/decision timestamps, spread/slippage, latency, terminal/session handling. | Full environment/backtest/live parity remains outside the bounded tests. |
| WP06 — metrics and intervals | Completed-position metrics, consistent trade counts, shared confidence-interval conventions, dependence-aware procedures. | Rebuild all campaign reports from raw position/fill data; do not accept stale aggregates. |
| WP07 — feature fit scope | Production fold builder, train-only fit scope, causal multi-timeframe alignment, cache scoping. | Preserve evidence for the actual campaign configuration and all fold boundaries. |
| WP08 — eligibility | Required gates fail closed; Idea 2 post-inversion latch, matching direction and HTF bias; inversion bar excluded; no required retest; inactive runs labelled. | No trained-policy funnel evidence yet; zero trades must remain an inactive result, not pass evidence. |
| WP09 — TP references | Causal target provenance, buffers, precision-aware deduplication, one/two/three target support, and approved mode semantics. | Confirm final traces/report exports preserve target provenance. |
| WP10 — matched comparison/reward attribution | Comparison helper checks matched non-intervention settings and reward components reconcile. | No provenance-complete treatment/control policy pair exists yet; no causal effect is established. |
| WP11 — provenance | Fail-closed manifest creation for source/config/data/cache/checkpoint/evaluation identity. | No certified historical `run_manifest.json` exists for old runs; never backfill as if contemporaneous. |
| WP12 — parity | Bounded deterministic traces and selected cross-engine comparisons pass. | No exhaustive production TradingEnv/broker/evaluation/backtest or live parity claim. |
| WP13 — handoff/docs | Source-truth records and validity documents updated. | Keep this master plan and current implementation aligned; do not imply unrun experiments in thesis text. |

**Approved trading contracts to preserve:** Idea 2 enters after IFVG creation
once direction and HTF bias align; it does not require a later retest and
cannot enter on the inversion bar. Long trailing stops use a confirmed
available swing low minus buffer; short stops use a confirmed swing high plus
buffer. Targets must be causal, executable, ordered, and traceable to their
reference/timeframe. Long active targets are above entry and strictly
increasing; short targets are below entry and strictly decreasing. Disabled
and hit slots stay disabled/hit; do not manufacture additional targets. The
approved TP modes are versioned: `fixed` never moves an entry target; legacy
`breakeven` makes one nearest-target pullback by configured alpha times initial
risk without moving the stop; `trailing` extends only the nearest remaining
unhit slot from a confirmed directional swing, effective on the next bar.
Reject crossed/overtaking updates. Partial fills conserve quantity, fees, net
PnL and reward exactly once; only full position completion ends the lifecycle.

### 8.2 Current 18-variant screen contract

The 18 names are the six-rung ladder applied to each strategy family:

| Family | Variants |
|---|---|
| Baseline | `ladder_a0_flat`, `ladder_a1_sl`, `ladder_a2_tp`, `ladder_a3_multi_tp`, `ladder_a4b_entry_fsm`, `ladder_a4a_entry_fsm_hidden` |
| Idea 1 — PO3/IFVG | `ladder_a0_flat_po3`, `ladder_a1_sl_po3`, `ladder_a2_tp_po3`, `ladder_a3_multi_tp_po3`, `ladder_a4b_po3_ifvg`, `ladder_a4a_entry_fsm_hidden_po3` |
| Idea 2 — distribution | `ladder_a0_flat_dist`, `ladder_a1_sl_dist`, `ladder_a2_tp_dist`, `ladder_a3_multi_tp_dist`, `ladder_a4b_distribution`, `ladder_a4a_entry_fsm_hidden_dist` |

All are PPO/TCN and VAE-free. The screening contract is seed 42, 20M PPO
timesteps per variant, `n_envs=64`, shared-bundle data, and full chronological
training-split episode mode. Each worker must complete the requested budget or
have a terminal failure reason, preserve its checkpoint and attempt lineage,
and produce deterministic in-sample train and report-only test evaluations.
The current screen is one-seed exploratory selection, not the final five-seed
comparison.

`ladder_a0_flat` is the pure legacy baseline with `strategy_actions=false` and
the legacy `Discrete(20)` action path. The other 17 ladder rows have different
action/feature/reward configurations. In particular, A0 versus A1 is not a
clean one-variable ablation; PO3/distribution families also differ from the
baseline family in strategy features, direction head, and reward. Use a
separately frozen matched pair for causal claims. Record the fully merged
effective config, post-scaling PPO values, action/observation dimensions,
reward/risk/cost settings, resolver overrides, source/dependency identity,
worker seeds, data/cache hashes, GPU/CPU environment, and checkpoint digest.

The training gate rejects a risk-limit breach or training peak trailing
drawdown above 10%. PnL, equity slope, and trade count are reported metrics,
not pass requirements; there is no 50-trade minimum. Evaluate the trained
checkpoint on both its full training split and following five-to-six-month
test split. The training evaluation is in-sample. The 2026 test interval has
already been inspected and is `selection_acknowledged`; it is report-only and
exploratory, never a pristine confirmatory holdout. Historical 100k pilot
train-equity failures remain failures and are not converted into passes by
this longer screen.

### 8.3 Data, multi-timeframe features, and prebuilt caches

- Primary series: US100.cash M1; secondary series: US500.cash M1. Existing
  source coverage is 2024-12-30 through 2026-07-03; the configured training
  split ends 2025-12-31 and the following test interval begins in 2026.
- Feature construction can combine M1 with synchronized M5, M15, and H1
  structure. Every higher-timeframe feature must be aligned causally to M1
  using forward fill only, never backward fill. A swing is usable only after
  its right-side confirmation bar is complete. Keep raw prices for execution;
  normalization is fit on the applicable training mask only.
- Baseline features, Idea 1 PO3/IFVG context, and Idea 2 distribution/PD
  context are distinct caches. The three prebuilt matrices are
  `data/prebuilt_feature_cache/baseline.parquet`,
  `data/prebuilt_feature_cache/idea1_po3_ifvg.parquet`, and
  `data/prebuilt_feature_cache/idea2_distribution.parquet`.
- `variant_resolver` selects those named caches only for their matching three
  A0 variants. `train_rl` verifies the `.hash` sidecar against effective
  feature config, raw market data, and training split. Idea caches also need
  `.htf_sweeps.parquet`. Missing files or mismatches fail closed; do not
  silently rebuild/overwrite these named caches. Other variants use the
  content-addressed cache path. These generated files are local data, excluded
  from Git, and must be present on any non-shared checkout/node.
- Shared-bundle parity passed exact tested traces for all 18 variants on the
  checked calendar, including zero-action and active-action traces. It does
  not prove exhaustive state-space parity. Re-run relevant parity after any
  feature, resolver, cost, reward, or environment change.

### 8.4 Throughput and GH200 optimization workstreams

The following are measurements with their boundaries, not claims of improved
learning quality or production campaign completion:

| Workstream | Evidence/status | Decision/remaining acceptance |
|---|---|---|
| Device-staged PPO rollout | Two paired GH200 PPO-core runs measured 3,404 steps/s versus 2,140 for CPU rollout storage: +59.1%, update time −62.8%, about +0.71 GiB peak VRAM; minibatch parity exact. Enabled on the fast branch with a CPU-buffer control. | PPO-core result only. Verify complete train/test quality before treating as a campaign improvement. |
| Batch/rollout geometry | Reference 128 steps/env, batch 4096, 15 epochs measured 3,519 steps/s and 960 optimizer steps/run. Batch 8192 at same geometry measured 3,697 steps/s and 480 optimizer steps; larger geometry reached about 3,813 steps/s but also changed optimizer/update count and used more VRAM. | No altered geometry selected; run controlled learning-quality comparisons before adoption. |
| Host synchronization | Deferred metric scalar reads had exact CPU parameter parity but repeats (3,537/3,568 steps/s) were within reference spread. | Production path removed; no repeatable gain established. |
| Pinned host prefetch | Summed copy time was about 134 ms across three PPO updates; full rollout staging already removes repeated minibatch transfers. | Not implemented; revisit only if an exact production profile shows material transfer cost. |
| TF32/BF16/FP8 | Existing TF32/BF16 retained. `medium` matmul measured 3,576 versus 3,581 steps/s at `high`; no gain. Full-policy BF16/FP8 not enabled. | Keep current precision absent a measured, numerically acceptable gain. |
| `torch.compile` and CUDA Graphs | Compile failed before training because GH200 runtime lacked a C compiler. CUDA Graph test not run. | Blocked pending a compatible toolchain and a measured learner-overhead case. |
| GPU-vectorized TradingEnv | On sharded backend, `TradingEnv.step()` averaged about 0.34 ms; rollout/framework phase was about 50.20 s versus max shard-summed env step 6.09 s. | Not justified by phase share; major rewrite deferred. |
| JAX/XLA | No project implementation; faithful port includes environment/execution semantics as well as PPO. | Deferred, separate high-risk research line. |
| Vector backend memory | A3 one-GH200 paired means: Subproc 2,249.5 steps/s, 10.90 GB ready cgroup RAM, 9.32 GiB PSS; sharded-16 2,193.2, 6.40 GB, 4.98 GiB. A separate nine-trainer paired sample measured sharded 7,983 vs Subproc 7,748 aggregate steps/s and materially lower RAM/PSS. | Sharded mode is opt-in. Small throughput edge is provisional; exact seeded behavior parity and campaign-level benefit are required. |
| Concurrency sweep | One-GH200 device-buffer PPO-core aggregate: T=1 3,429; T=2 5,230; T=3 6,581; T=4 7,137; T=6 8,003; T=9 8,699 steps/s. T=9 with 2-second start stagger reached 8,924 steps/s; single update-slot semaphore reduced throughput. | T=9 stagger is a benchmark candidate, not production validation. Candidate 18-worker/2-GH200 PPO-core time is ~5.6h before callbacks, evaluation, export, and startup. |

The current live 18-run progress (321–575 per-worker FPS) is a separate,
full-campaign observation and must not be directly compared with short
single-GH200 PPO-core probes. The historical 18-worker 100k screen projected
18.4–42.8h per 20M worker from a different frames/`n_envs=4` profile; it is
not the current run's forecast. Report aggregate completed steps/s and
end-to-end makespan, not per-worker FPS alone.

**Required remaining speed sequence:** after the live campaign/supervisor is
terminal, repair the benchmark so it consumes the exact resolved campaign
config; measure startup/cache build, reset/features, rollout and IPC, PPO
update, callbacks/dashboard, checkpoints, train/test evaluation, and export
separately. Compare 1/3/6/9 trainers per GH200 and `n_envs` 16/32/64 only with
matched source/data/config, fixed PPO semantics, repeated windows, and valid
resource samples. Record per-worker and aggregate FPS, CPU use/throttling, GPU
use/VRAM, allocation-scoped cgroup RAM/peak/OOM, process PSS, disk I/O, worker
count, and walltime. Select the reliable maximum aggregate completed
training-steps/hour while preserving behavior and survival gates.

### 8.5 Shared-bundle, sharding, RAM, and controller rules

- Bundle data are immutable/read-only memory maps. `n_envs` counts logical
  PPO environments; `vector_shards` counts physical subprocesses. Shards are
  contiguous and balanced; each logical environment retains reset seed,
  state, terminal observation, and episode boundary. `sharded_bundle` requires
  `env.data_source=bundle`; default `subproc` behavior is unchanged.
- Historical 18-worker bundle probe: `n_envs=64`, six CPUs/worker, two
  GH200s; PSS averaged 218.9 GiB and peaked 251.2 GiB, GPU use averaged 66.4%
  with sampled peaks 97%/96%. This is bounded probe evidence, not 20M
  allocation peak sizing. Capture cgroup memory at the job allocation scope;
  do not sum shared RSS or use sampler-step memory as job memory.
- Full bundle acceptance requires matched frames/bundle comparisons across
  representative baseline, PO3, distribution, action widths, VAE and
  walk-forward cases where used. Compare every transition: observations,
  rewards and components, NAV/equity, positions/fills/fees, masks, diagnostics,
  termination/truncation and causes, with fixed seeds and declared float
  tolerances. A PPO parameter hash or short selected trace is support evidence,
  not comprehensive policy equivalence.
- Measure peak allocation memory and require at least 64 GiB reserve plus
  1.35× predicted peak within the job cgroup before admitting a workload.
  Keep financial precision; any downcast needs explicit parity review. Do not
  shrink market history, disable evaluation, or change PPO math to save RAM.
- Controller run identity must include variant/seed/budget/source/config/data
  hashes. Attempt directories and append-only events preserve stderr, exit
  codes, artifact hashes, checkpoint validity, host/job/GPU, and failure
  reason. Resume must validate provenance and avoid duplicate scheduling or
  overwriting successful outputs. A nonzero exit caused by a deterministic
  research gate remains a visible failure; do not retry just to hide it.
- On Triton reuse the existing allocation/supervisor first. New `srun` and
  `scancel` require explicit user authorization. Current project rules limit
  allocations to at most two GPUs; one model uses one GPU. Keep one persistent
  compute-node tmux with durable campaign/controller/worker/resource logs.
  GH200 uses aarch64 `.venv`; x86 H100/H200 uses `.venv-x86`, selected by
  `scripts/triton/activate_venv.sh`. Login nodes are for read-only queue,
  tmux, and file checks, not training or `n_envs=64`.

### 8.6 Evaluation and report generation

Evaluation speed means time from checkpoint to valid numeric result and time
from checkpoint to all requested visuals; report those separately.

1. Preserve the historical observed example as a warning about export cost:
   one A0 attempt took about 10 minutes for numerical test/train evaluation,
   then roughly 81 minutes for export, including hundreds of per-trade PNGs.
   It is one run's timestamp-derived evidence, not a repeated profiler result.
2. Keep `full` as the complete backward-compatible report profile. The
   implemented `campaign_fast` path retains train/test inference, risk/equity
   metrics, gates, checkpoint, provenance, machine-readable reports and CSV;
   only explicitly optional per-trade images/HTML may be deferred. A
   deterministic `plots_only` postprocessor should render from saved
   trades/equity/bars/config without retraining or changing numerical results.
3. A report profile must never skip train/test inference, alter policy
   decisions/RNG, omit survival outcomes, convert `TRAIN_EQUITY_FAILED` to
   success, erase checkpoints, or change controller research status. List
   deferred visuals in a manifest. Plot failures are visualization failures;
   numerical/gate failures remain authoritative.
4. Before changing dashboard cadence or isolating its stochastic policy
   predictions, preserve and restore Python, NumPy, Torch CPU, and CUDA RNG
   state. `log_every` also updates an active prior, so do not reduce it merely
   to quiet logs.
5. Remaining work: collect per-phase `perf_counter` timing, chart counts,
   bytes, failures, CPU/PSS and filesystem throughput; reuse indexes and
   plotting context; close figures; use Agg; test bounded spawn-safe chart
   concurrency only if measured. Benchmark safe node-local scratch followed
   by verified atomic publish; never leave unique required artifacts only on
   ephemeral storage.
6. Acceptance: use the same saved checkpoint/data/seed/config for old/full and
   optimized reports. Check exact or justified-tolerance identity of actions,
   trades, fills, SL/TP, equity, rewards, risk, metrics, and gates. Verify
   chart parity on long/short/partial/risk-terminated/FSM cases, resumability,
   idempotency, atomic publication, memory bounds, and visible failures. Aim
   for an 80% reduction in checkpoint-to-numeric-report time, but do not claim
   it until at least three matched measurements support it.

### 8.7 Quality program: implementation state and exact registered stages

The quality program is a separate lineage from the 18-rung screen. Its
registry uses winner parents that fail closed until a previous stage winner is
explicitly selected. Code availability is not training evidence. Current
foundations: versioned account-state v1/v2, metadata validation, v2 on frame
and bundle paths, optional diagnostics, per-step constraint telemetry, Q4A
semantic Gaussian heads, and reproducible PPO/vector-worker seed wiring. The
default observation stays v1. `equity_delta_r` requires fixed-risk sizing and
rejects reward-shaping combinations; its telescoping identity does not apply
to hard-breach terminal penalties. Q4B hybrid distribution and Q5
PPO-Lagrangian (cost critic, cost buffer, dual update, or learned shield) are
not implemented. No Q variant has campaign-quality evidence or a selected
winner.

| Gate | Registered experiment | Required comparison and status |
|---|---|---|
| Q0 | `q0_control_a3`, account-state v1, A3 multi-TP control | Freeze config/data/seed/budget/checkpoint/evaluation identity; establish deterministic full-split train/test control before Q1. Not run. |
| Q1 | `q1_state_v2` | Change only account observation schema versus Q0; v2 implementation/unit tests exist, model comparison not run. |
| Q2 | v2 with `n_envs` 16, 32, 64 | Compare rollout geometry with all other semantics fixed; select only after throughput and learning-quality review. Not run. |
| Q3 | RR/fixed-risk $500; RR/fixed-risk $1,000; equity-delta-R/fixed-risk $500 | Compare to Q2 winner one reward/risk definition at a time; freeze whether these are three arms or a staged subset before launch. Not trained. |
| Q4 | semantic heads; hybrid categorical-continuous distribution | Q4A code exists and is opt-in; Q4B is not implemented. Compare one action-distribution change against Q3 winner and verify action/probability semantics. |
| Q5 | Lagrangian constraints | Not implemented beyond telemetry; first implement and verify cost critic/buffer/dual protocol, then compare against Q4 winner. |
| Q6 | visible FSM versus no-FSM control | Change entry-state visibility/handling only; preserve after-IFVG semantics and match all other settings. Not trained. |
| Q7 | LiteTCN versus TCN | Compare encoder only against Q6 winner; keep data/budget/seeds/evaluation equal. Not trained. |

All quality experiments need at least the user's one-year training exposure
before deterministic training/test evaluation. Use one discovery seed first;
five paired seeds follow only after a valid full-split result and an
authorized comparison. PnL/trade count remain reported measures, not hard
gates. Do not run stages out of parent order or select on the previously
exposed 2026 interval.

## 9. What “one plan” means operationally

Read this file for the full current roadmap, technical contracts, status,
acceptance criteria, sequencing, resource boundaries, and planning estimate.
Use the append-only implementation ledger and immutable campaign artifacts
only when a stage asks for verification evidence; they are evidence records,
not extra plans. The reviewer checklist is the required form for recording
independent review, and the experiment YAML is the Q0–Q7 variant registry; this
document states how and when to use them. All other listed plans are retained
for audit/history and are not additional queues. Update this master plan when
stage status or author decisions change, rather than creating another plan.
