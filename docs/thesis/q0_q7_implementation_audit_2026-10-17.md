# Q0–Q7 Implementation Audit — 2026-10-17

## Scope and conclusion

This audit checks the required Q0–Q7 quality-program stages against the
registered variants, trainer/model/environment wiring, focused tests, and
available validation notes. It is an implementation audit, not a campaign
result, strategy evaluation, winner selection, or thesis-readiness decision.

## Current-tree validation — 2026-10-17 11:18 Helsinki

- On MI210 job `21014127`, the final source tree passed full-tree Ruff format,
  Ruff lint, and Mypy (464 files); a real ROCm 6.3.1/PyTorch 2.7.1 MI210 matrix
  operation passed; and all **1,682 tests passed with 77 warnings** in 3:41.
  After formatting, the directly relevant environment, observation, evaluation,
  numerical-report, comparator, registry, and Lagrangian tests passed again:
  **112 passed, 37 warnings**. Log:
  `outputs/gpu_ci_gates/q0_q7_current_tree_job21014127/validation.log`.
- The first targeted run caught a defect: legacy-sequence shared bundles
  received the same key as standard bundles because the root-level mode flag
  was absent from `shared_bundle._RELEVANT_CONFIG_PATHS`. The key now includes
  `legacy_sequence_columns`; the regression and frame/bundle parity tests pass.
- The deterministic reanalysis comparator now verifies that action,
  observation, stop-candidate, sequence-feature, and backend contracts match
  the source run manifest. For a source lacking sequence coordinates, it
  requires the explicit legacy disposition and opt-in. Cache identity,
  source datasets/checkpoint/config, evaluator/runtime identity, legacy modes,
  and raw numerical artifacts are also checked by the comparator.
- This gate validates source behavior on the MI210 runtime, not equivalence to
  the thesis's locked CUDA/GH200 trainer, data provenance, Q-stage training,
  policy quality, or thesis readiness. Q2/Q3/Q4/Q6 remain registry
  `implementation_unverified` and therefore fail closed; Q5 is selectable only
  after its Q4 parent is frozen. The Q stages have no completed research run or
  winner.
- The same approved allocation completed the P1 raw-tick audit. The job-specific
  split-aware report is
  `docs/thesis/g1_tick_quality_current_recovery_2026-10-17_job21014127_v2.json`
  (SHA-256 `6db8e4a88c13dec284ca357039c32f10cdf5614929f865d280e451061f4cae5a`).
  It verifies campaign input hashes and records 100% test-period minute
  coverage, but only 1.16% US100 and 99.82% US500 coverage over training.
  This limits the existing strategy/cost evidence and does not provide feed
  rights, broker timezone/DST, fill-source parity, or an acceptance threshold.

Q0 must not be conflated with the thesis's pure-legacy reference baseline.
`q0_control_a3` is the account-state-v1/A3 development control used inside the
quality program; the thesis's pure-legacy A0 reference is `ladder_a0_flat`
(`strategy_actions=false`). They answer different comparison questions. Keep
the registered Q0/Q1 quality-program contrast distinct from the frozen thesis
control/treatment unless a later protocol explicitly aligns them.

## Addendum — source inventory refresh, 2026-10-17

### Current-tree software gate — 2026-10-17 08:58 Helsinki

- MI210 job `21012389`, step `.14`, passed full-tree Ruff format/lint,
  full-tree Mypy across 461 files, an actual MI210 ROCm matrix operation, and
  the complete suite: **1,671 passed, 77 warnings**. The validated tracked
  worktree diff SHA-256 is
  `a52a452848e99f68232da7240ea0376490e549dd9770583d854333ab5a56a360`;
  full log: `outputs/gpu_ci_gates/observation_schema_gate_job21012389/validation.log`.
  This validates current code and tests, including the tick-ingestion and
  observation-schema changes, but does not establish market-data provenance,
  full-campaign causal validity, evaluator acceptance, strategy quality, or
  thesis readiness.
- The full-training-interval fold-fit report completed at 08:18 and took
  3,758.9 seconds. Its hashed `quant_rl/features/build.py` is
  `7df39268c985920d97aaddefeef3f3da9c96a58de04b9575375e61e7bda212af`; the
  current source hash is
  `256387fc132498d9846f1c17526ea19a900d90def012703b90582c92c44fdcc1`.
  The 20/20 report remains valid for its recorded source snapshot but is not
  current evidence for the final observation-column filter. The remaining
  MI210 allocation window was shorter than the measured rerun duration, so no
  repeat was launched. P1 remains open pending a suitable execution window
  and subsequent real-data/cache audits.

### Latest validation follow-up — 2026-10-17 06:45 Helsinki

- A separate authorized MI210 allocation, job `21012957`, started on
  `gpuamd1` with 8 CPUs, 96 GiB, and one MI210. A real ROCm matrix operation
  succeeded. This ad-hoc attempt ran Ruff successfully but invoked Mypy from
  `.venv-mi210`; the canonical GPU gate uses `.venv-x86` for Ruff and Mypy.
  The three resulting errors in `ppo_policy.py` and `hybrid_actor.py` are
  environment-specific and do not contradict the canonical gate. No source
  changes from that attempt remain; the temporary annotations were reverted.
  The allocation ended after 2m25s and its logs remain in
  `outputs/gpu_ci_gates/q0_q7_full_validation_job21012957/`.
- The canonical full-tree gate at 06:27 in job `21012389`, step `.9`, passed
  Ruff format/lint, Mypy from `.venv-x86` (460 files), the MI210 smoke, and all
  1,664 tests. The wrapper then exited nonzero only because summary generation
  refused to overwrite existing immutable outputs. The exact tracked-source diff hash
  `c76ea5d3ddf72497f1c1375b43666d8f358fb570176884daaa63a220ded2066a` and
  separately recorded hash for the then-untracked report verifier and test
  `43a92db7b1f63fe6d73f606b147b5eba553a1d5773d711f1dbc38a0955ac43a9` match
  the now-committed code. The wrapper's only failures were safe refusals to
  overwrite existing immutable campaign summaries. The validated source was
  committed as `a611411`; later commits `bc3e78f` and `953a60a` were
  documentation-only at that point. Subsequent dirty-tree P1 tick-ingestion
  and observation-schema changes invalidate the claim that this run verifies
  the current source; a current-tree GPU gate is still required.

### Prior implementation validation — 2026-10-17 06:31 Helsinki

- On MI210 allocation `21012389`, before the later P1 tick-ingestion and
  observation-schema edits now present in the dirty worktree, full-tree Ruff
  format, full-tree Ruff lint,
  full-tree Mypy (460 files), a real ROCm matrix smoke, and the full test suite
  passed on that earlier source snapshot: **1,664 passed**, 77 warnings. The complete
  log is `outputs/gpu_ci_gates/p2_feature_cache_reanalysis_20261017/current_tree_validation.log`;
  the validated source diff hash recorded there is
  `c76ea5d3ddf72497f1c1375b43666d8f358fb570176884daaa63a220ded2066a`.
- The wrapper's final report-generation phase exited nonzero because its
  immutable-output guard correctly refused to overwrite existing schema-v2
  summaries. Both reports already exist, declare seed 50, and embed the
  current summary-script SHA-256 `74b365f989bc7d841082c0e7a6a6340dddce76e1a80ecbfbe5ee724b73fd15e8`:
  recovery report SHA-256 `aeeb5eabab0fd568b619f368d4bb937265dbfd6f19bc32f1f85849af5cadbda7`,
  v3 report SHA-256 `edbd0312b473d118982a68b6c38ebff2e1c5d0b254092e8a727e63b4751355c1`.
  The original schema-v1 artifact remains preserved. The software gates pass;
  the wrapper is **not fully green** because those immutable reports were not
  regenerated. No report was overwritten.
- Q5's objective mismatch is repaired: policy penalty and dual budget now use
  the same gamma-discounted episodic cost. Version-2 constrained checkpoints
  retain gamma; resume without restored environment/account state fails closed.
  Deterministic evaluation exports per-step and component costs, breach
  indicators, discount factors, and a summary; the numerical-report verifier
  reconciles these fields. Tests cover callback objective, checkpoint state,
  evaluator export, and verifier. This validates bounded software behavior
  only, not a learned-policy result or formal risk guarantee.
- Allocation `21012389` remains RUNNING on `gpuamd1` (MI210 x1, 8 CPUs,
  128 GiB; scheduled end 09:26 Helsinki). No campaign/training was launched.

- The one-minute allocation wait/fallback rule is already present in
  `.agents/rules/triton-slurm.md`: inspect each candidate node's GPU, CPUs, and
  RAM, cap resource polling at 60 seconds, preserve pending requests unless
  separately authorized to cancel, and consider compatible A100-or-better or
  MI210 alternatives. It needs no duplicate rule.
- The registry test now asserts the full mandatory stage inventory and Q0
  control identity, so deleting or renaming a required stage fails validation.
  Q0 is the control arm inside the Q1 state-comparison stage; the registry has
  seven stage nodes for the eight Q0–Q7 arms.
- Q2 fixes its common aggregate PPO rollout at 8,192 samples across 16/32/64
  environments using per-environment `n_steps` of 512/256/128, binds each
  variant to its registered exact worker count, and skips automatic PPO
  geometry scaling while checking the host-memory cap. The trainer validates
  the actual `n_envs * n_steps` product. The correction was made at 05:38
  Helsinki after finding the previous `n_steps=8192` values were per
  environment, yielding 131,072/262,144/524,288-sample rollouts. The correction
  and tests passed the earlier MI210 gate, not the current dirty-tree gate.
- Q5 constructor validation rejects non-finite budget, dual-rate, and
  multiplier-bound settings. Discounted objective, resume guard, evaluation
  cost export, and report reconciliation tests also passed. This does not make
  the method a hard risk guarantee or make Q5 launchable without Q4.
- Q0–Q7 remain compulsory in the overall program, but this does not make their
  research runs complete or launchable before their parent and validity gates.
  In the registry, Q2/Q3/Q4/Q6 arms remain `implementation_unverified`; Q5 is
  implemented and unit-tested but still depends on a frozen Q4 parent; Q7 has
  selectable encoder implementations but no comparative result. No stage has a
  winner or a Q-stage campaign result.
- The RQ1 stop-candidate code audit confirms M5/M15/H1 multi-candidate feature
  generation, completed-bar confirmation times, sweep removal, price-level
  deduplication, source-TF age, causal stop-menu filtering, deterministic
  distance ordering, and candidate audit provenance. The thesis profile requests
  three candidates per side; the default remains one legacy candidate. The
  compatibility schema now also records configured encoder timeframes merged
  into the effective stop menu, with a custom-M30 regression test.
- The audit found that cache identity already included feature configuration,
  but the run manifest/checkpoint did not explicitly identify the stop-menu
  contract. Code now adds an `rq1-stop-menu-v1` record to both, rejects resume
  under a missing or changed checkpoint contract, and makes reanalysis of older
  checkpoints require `--allow-legacy-stop-schema`. The explicit legacy path
  does not claim the expanded menu was used. Focused tests and the GPU-node CI
  gate passed on the earlier dirty-tree snapshot; the later P1 observation-
  schema edits have not yet passed a GPU-node gate. See Verification boundary.

**Q0–Q7 are compulsory in the master plan, but the research program is not
complete. No Q-stage campaign has run and no stage winner is frozen.** Most
stages have registered source arms and bounded unit coverage, but source
presence does not make them empirically complete or ready to launch. Q5's
previous registry placeholder has now been implemented as an opt-in,
primal-dual scalarized-return PPO objective. An earlier source snapshot passed
full-tree Ruff format/lint, Mypy, MI210 smoke, and 1,664 tests; the later
current-tree gate on job `21014127` passed Ruff format/lint, Mypy, MI210 smoke,
and 1,682 tests. The policy and
dual update use matching gamma-discounted costs; Q5 evaluation costs are
exported and independently reconciled, and policy-only resume fails closed.
The CI wrapper exited 1 only because it safely refused to overwrite existing
immutable schema-v2 campaign summaries. Q5 remains blocked on freezing the Q4
parent and has not been run as a research experiment.

## Stage-by-stage status

| Stage | Implemented in source | Blocking work before valid research run |
|---|---|---|
| Q0 | `q0_control_a3` registers the A3/account-state-v1 control and quality diagnostics. | Freeze full data/config/code/seed/budget provenance; run complete chronological training and deterministic train/test evaluation; package all outputs. No Q0 result exists. |
| Q1 | Account-state-v2 schema, metadata validation, frame/shared-bundle wiring, and focused tests exist. | Revalidate the current final integration, then run a matched Q0/Q1 pair. Report behavior and economic metrics separately. No Q1 result exists. |
| Q2 | The 16/32/64 arms declare per-environment `n_steps` 512/256/128 to hold aggregate rollout samples at 8,192; exact environment count is pinned, hardware PPO scaling is bypassed, and insufficient host-memory capacity fails closed. Current-tree full tests (1,682), Ruff, Mypy, and MI210 smoke pass. Registry arms still fail closed as `implementation_unverified`; no geometry comparison or winner exists. | Complete implementation/integration validation, freeze minibatch/update count, epochs, wall-clock/resource accounting, and parent checkpoint on one locked accelerator; then run and review the geometry comparison. |
| Q3 | RR, fixed-risk, and `equity_delta_r` reward/risk code exists; `fixed_risk_usd` is a supported environment setting. | Validate quality-registry-to-trainer resolution and reward diagnostics; freeze one Q2 parent and one multi-arm comparison family. Keep reward return separate from PnL/equity. No run or winner exists. |
| Q4A | Semantic Gaussian actor heads are wired into PPO and preserve the declared action-coordinate order. Focused tests cover distribution behavior, gradients, and checkpoint/schema behavior. A bounded MI210 semantic-head PPO update and save/load check is recorded in `docs/operations/triton_environment.md`. | Current registry arm is fail-closed. Revalidate on the campaign's locked CUDA stack and verify deterministic evaluator parity from the selected Q3 parent. No Q4A research run exists. |
| Q4B | Hybrid categorical/Gaussian actor and PPO policy wiring exist; focused test source covers sampling, log-probability, entropy, gradients, and save/load. A bounded MI210 hybrid PPO update and save/load check is recorded in `docs/operations/triton_environment.md`. | It is opt-in and requires the shared actor architecture, a declared overlay `ActionSchema`, and an active exit coordinate when `exit` is present. Validate every category center against the environment decoder and campaign-locked trainer/evaluator/checkpoint integration. Registry arm remains fail-closed; no Q4B research run exists. |
| Q5 | Opt-in primal-dual scalarized PPO uses the same gamma-discounted episodic composite risk cost in policy updates and dual-budget comparisons; schema-v2 checkpoint state stores gamma and refuses policy-only resume without restored environment/account history. Deterministic evaluation exports per-step/component costs, breach indicators, and discount factors; the numerical-report verifier reconciles them. Current-tree full tests (1,682), Ruff, Mypy, and MI210 smoke pass. Hard account guardrails are unchanged. | Freeze Q4 parent, budget/risk-cost interpretation, and evaluation protocol; independently review the method; then run the complete comparison. No Q5 result exists; implementation tests do not validate a constrained policy or formal risk guarantee. |
| Q6 | Entry state machine, hidden/visible observation switch, no-FSM control, diagnostics, and three-arm registry exist. | Revalidate transition and observation semantics in both frame and shared-bundle paths. Preserve the three-arm design: no FSM, hidden FSM, visible FSM. Freeze the Q5 parent before a run. No Q6 result exists. |
| Q7 | TCN and LiteTCN are selectable; LiteTCN has encoder and agent/checkpoint tests. | Validate matched trainer/evaluator/checkpoint integration and use the frozen Q6 parent, data, action/reward/risk semantics, budget, and seed plan. No Q7 result or winner exists. |

## Launch and ordering controls

- The registry in `config/quality_experiments.yaml` is separate from historical
  experiments and does not auto-launch any campaign.
- `apply_quality_variant()` refuses unresolved `*_winner` parents and refuses
  variants whose status is not explicitly selectable. Keep this fail-closed
  behavior; promote a variant only after its integration checks pass.
- Run Q stages in the dependency order in
  `docs/thesis/master_project_execution_plan.md`; record a winner/no-winner
  disposition and freeze the next parent before opening its child stage.
- The user has made the Q program compulsory. Its planned timing remains after
  the thesis submission unless explicit authorization and entry gates allow
  earlier work that does not displace thesis-critical tasks.
- No missing result may be replaced with a source audit, synthetic test,
  throughput measurement, or a result from the contaminated 18-variant screen.

## Slurm agent rule check

`.agents/rules/triton-slurm.md` already enforces the user's requested one-minute
behavior: inspect each candidate node's GPU, CPU, and host RAM before each
allocation attempt; use an immediate 60-second limit; stop active polling once
the resource is not immediately available; preserve pending jobs unless
separately authorized to cancel; and compare compatible alternatives no lower
than A100 or MI210 using fresh per-node inventory and `srun --test-only`.
This is a wait/poll policy, not permission to cancel or overlap an allocation.
The current user-provided `AGENTS.md` agrees. No duplicate or weaker rule was
added.

## Source-level implementation boundary refresh — 2026-10-17 11:03 Helsinki

The registry and trainer guards were re-read against the Q-stage table. The
registry has seven stage records because Q0 is the control arm inside the Q1
state-comparison stage; it still has the full required arm inventory. The
resolver allows only `control`, `implemented_unit_tested`, and
`existing_encoder_implementation` statuses. Q2, Q3, Q4A/B, and Q6 arms remain
`implementation_unverified` and fail closed; Q7's LiteTCN is present but still
depends on a frozen Q6 winner. Source presence or unit coverage does not make
these arms selectable, integrated, or empirically complete.

Q5's Lagrangian line is implemented in source and has bounded callback,
checkpoint, evaluation-export, and verifier regressions in the earlier
validated source snapshot. Its current dirty-tree changes are not validated.
The implementation uses the maximum of the three per-step normalized risk-cost
increments (daily, max-loss, trailing) as one scalar cost, discounts that cost
through completed full-split episodes, and performs projected dual ascent
against one scalar budget. It is a scalarized primal-dual experiment, not three
independent constraints and not a formal risk guarantee. It rejects MVP,
risk-terminated and policy-only-resume cases. No Q5 run exists, and a Q4 parent
must be frozen before Q5 can be selected.

Q2's registered `n_envs` arms preserve 8,192 aggregate rollout samples using
512/256/128 steps per environment and fail closed on resource scaling that
changes the declared count or aggregate rollout. That software contract does
not yet freeze minibatch/update count, wall-time accounting, or a measured
quality/throughput winner. Q4B's hybrid actor has source/tests but remains
fail-closed pending action-center/environment-decoder and full checkpoint,
trainer, and deterministic-evaluator integration review. The full per-stage
implementation and missing-evidence matrix is in the table above; no Q-stage
campaign, stage result, or winner exists.

The deterministic reanalysis comparator was also tightened to verify the
actual evaluated feature-cache file hash and identity against the original
training cache, copied source-manifest identities, complete evaluator/runtime
identity, and explicit legacy-mode flags, and reject legacy/unverified cache
dispositions. Its report records both evaluated cache paths and hashes.
Regression fixtures cover detached cache identity, cache tampering, source
dataset-manifest drift, evaluator-code identity, and source-schema mismatch.
The focused suite and full current-tree GPU-node suite now pass, including this
comparator. As of the 11:18 checkpoint, no reanalysis package has yet been
generated or compared, so P2/P3 report acceptance remains open.

## Observation-schema issue found during P1 audit review

The completed 20/20 fold-fit report exposed a separate model-input issue:
absolute event timestamps and identifiers such as `po3_ifvg_time_ns`,
`po3_ifvg_fvg_origin_time_ns`, and `sweep_high_pivot_time_ns` were included in
the numeric observation tensor. Those values are Unix nanoseconds or absolute
bar/setup IDs, not meaningful scaled state; the report's very large raw
validation deltas are driven by these metadata fields. They do not indicate a
price-feature magnitude or by themselves establish future leakage.

The current working tree filters absolute timestamp, origin-bar, and setup-ID
suffixes from observations, keeps relative age/state fields, and bumps
shared-bundle version from 2 to 3 so stale observation tensors cannot be reused.
Training manifests now record ordered sequence-feature coordinates and a hash;
evaluation compares that contract and checkpoint sequence width, with an
explicit opt-in for legacy manifests. Regressions cover bare and timeframe-
prefixed metadata, order-sensitive schema hashes, bundle-key invalidation, and
frame/shared-bundle observation parity with absolute metadata present. These
current-tree regressions passed the full MI210 software gate at 08:58 (1,671
tests, Ruff format/lint, full-tree Mypy, and ROCm smoke). The separate real-data
fold-fit report still hashes the earlier `features/build.py` source; its 20/20
result is not a post-filter full-interval audit. Archived checkpoints and the
prior contaminated campaign retain their old layout; they are not retroactively
corrected by this software change. Future runs must train and evaluate under
the versioned observation schema. The consolidated full-tree Ruff/Mypy, MI210
ROCm-kernel smoke, and full-pytest runner is
`scripts/research/run_observation_schema_gate.sh`. Before the gates, the runner
records the HEAD, git status, tracked diff SHA-256, and hashes of
untracked files to bind validation to the dirty source snapshot. Q0–Q7 remain
compulsory and empirically unrun.

## Verification boundary

### Real-checkpoint inventory for the next P2 replay

- The sole bounded-clean profile, `ladder_a0_flat`, has a saved
  `ppo_final.zip` (SHA-256
  `1d7e9f7e678b8240c22935cc23eeb419928803578544499a7d037043c72a8879`) and
  manifest, but the recovery report records the train-equity stop
  `max_loss_stopped_trading`, zero completed full-split episodes across its 64
  environments, and no completed annual train episode. It is not a completed
  training-policy acceptance candidate.
- `ladder_a4b_distribution` is the most complete available replay candidate:
  its run manifest records 20,003,008 aggregate steps, 64/64 complete 99,504-step
  training-split episodes, checkpoint SHA-256
  `7d81739de7e1bc54d818149a0bacd6525e58d1e06b4d7e88c2b060dc55f21ba3`, and an
  exact source-cache identity/file hash. Its report-only test traded on only
  two of 130 days; the source configuration is among the 17/18 profiles with
  the pre-repair NY-session lookahead, and its run manifest lacks the current
  versioned stop-candidate schema. A deterministic replay of it can test
  evaluator behavior under an explicitly legacy schema opt-in. The numerical
  verifier correctly will not accept that legacy report as canonical P3
  evidence. It cannot support selection, close P2's completed-selected-checkpoint
  criterion, or establish a successful policy.
- Thus no available checkpoint from this screen can close P2/P3 end to end:
  the baseline is clean but did not complete training-split exposure, while the
  complete candidate is scientifically contaminated and schema-legacy. A
  future replay is narrowly engineering evidence, clearly labelled exploratory.

The deterministic-replay follow-up now adds the feature parquet's file
SHA-256 to newly generated training manifests, as required by the verified
reanalysis path. A focused run-manifest regression asserts this field. The
current implementation and regression passed the full 1,671-test MI210 gate;
this does not retroactively upgrade historical manifests.

The P3 numerical verifier also checks that the saved equity curve has one
initial point plus exactly one point per recorded evaluation step, and now
reconciles Q5 cost exports. These regressions passed in the current 1,671-test
MI210 source snapshot recorded above. This validates software behavior only;
it does not produce an admissible P3 report or close P1/P2.

An earlier source snapshot was validated on MI210 job `21012389`, step 3, at
HEAD `599741fe710fa8b342ef3797448132413c07e4db`. The source diff for the
validated implementation files had SHA-256
`10367a0c445ee6005ac8923464b85b3a810d0dc7cd402d7ce864359cb0ab445b`. Full-tree
Ruff format and lint passed; full-tree Mypy passed for 458 source files; a real
ROCm 6.3.1 / PyTorch 2.7.1+rocm6.3 MI210 matrix operation passed; and all 1,603
tests passed with 38 warnings. The full log is
`outputs/gpu_ci_gates/q0_q7_current_tree_20261017_job21012389/current_tree_validation.log`.
That earlier snapshot is superseded for software validation by the later
1,664-test source snapshot recorded above. Both historical results remain
evidence only for their respective source boundaries.

At 05:19 Helsinki, the registry guard was extended to validate every declared
quality override against the canonical default-config path and value type,
then to reject unknown paths when applying a selectable lineage. This closes a
concrete silent-configuration failure mode caused by `force_add=True`, which
could create a typo key that the trainer never reads. Its regression test
passed in the earlier 1,664-test source snapshot and remains part of the pending
current-tree validation.

Earlier on 2026-10-17, H200 job `21004765` on `gpu59` (H200 x1, 48 CPUs,
512 GiB) passed CUDA matrix smoke, full-tree Ruff format/lint, full-tree Mypy,
and 1,594 tests (`gpu_ci_final3.log`). The initial MI210 validation in job
`21012389` also passed 1,594 tests (`mi210_full_validation.log`); step `.3`
later passed the recorded source snapshot with 1,603 tests. No training run,
checkpoint mutation, cancellation, Q-stage campaign, or stage promotion
occurred. Slurm's one-minute allocation wait rule and A100-or-better/MI210
fallback are already present in `.agents/rules/triton-slurm.md`; the rules
index now summarizes these requirements so agents see them at entry. The rule
was clarified to require the allocation attempt itself to return within 60
seconds, with no repeated polling/resubmission after an unavailable result.
