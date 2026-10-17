# Q0–Q7 Implementation Audit — 2026-10-17

## Scope and conclusion

This audit checks the required Q0–Q7 quality-program stages against the
registered variants, trainer/model/environment wiring, focused tests, and
available validation notes. It is an implementation audit, not a campaign
result, strategy evaluation, winner selection, or thesis-readiness decision.

## Addendum — source inventory refresh, 2026-10-17

- The one-minute allocation wait/fallback rule is already present in
  `.agents/rules/triton-slurm.md`: inspect each candidate node's GPU, CPUs, and
  RAM, cap resource polling at 60 seconds, preserve pending requests unless
  separately authorized to cancel, and consider compatible A100-or-better or
  MI210 alternatives. It needs no duplicate rule.
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
  gate now pass on the current dirty tree; see the exact run record in the
  Verification boundary below.

**Q0–Q7 are compulsory in the master plan, but the research program is not
complete. No Q-stage campaign has run and no stage winner is frozen.** Most
stages have registered source arms and bounded unit coverage, but source
presence does not make them empirically complete or ready to launch. Q5's
previous registry placeholder has now been implemented as an opt-in,
primal-dual scalarized-return PPO objective. Its unit tests, checkpoint round
trip, full-tree Ruff/Mypy, and current-tree MI210 suite pass (1,603 tests; see
verification below). The earlier H200 suite passed 1,594 tests. Q5 remains
blocked on freezing the Q4 parent and has not been run as a research experiment.

## Stage-by-stage status

| Stage | Implemented in source | Blocking work before valid research run |
|---|---|---|
| Q0 | `q0_control_a3` registers the A3/account-state-v1 control and quality diagnostics. | Freeze full data/config/code/seed/budget provenance; run complete chronological training and deterministic train/test evaluation; package all outputs. No Q0 result exists. |
| Q1 | Account-state-v2 schema, metadata validation, frame/shared-bundle wiring, and focused tests exist. | Revalidate the current final integration, then run a matched Q0/Q1 pair. Report behavior and economic metrics separately. No Q1 result exists. |
| Q2 | The 16/32/64 environment arms are registered. Trainer code adjusts per-env PPO horizon and logs logical env count, aggregate rollout size, minibatches, epochs, and updates per rollout. | `n_envs` alone does not define a fair comparison. Freeze total samples, `n_steps`, minibatches/update count, epochs, wall-clock/resource accounting, and the parent checkpoint. Arms remain fail-closed; no geometry comparison or winner exists. |
| Q3 | RR, fixed-risk, and `equity_delta_r` reward/risk code exists; `fixed_risk_usd` is a supported environment setting. | Validate quality-registry-to-trainer resolution and reward diagnostics; freeze one Q2 parent and one multi-arm comparison family. Keep reward return separate from PnL/equity. No run or winner exists. |
| Q4A | Semantic Gaussian actor heads are wired into PPO and preserve the declared action-coordinate order. Focused tests cover distribution behavior, gradients, and checkpoint/schema behavior. A bounded MI210 semantic-head PPO update and save/load check is recorded in `docs/operations/triton_environment.md`. | Current registry arm is fail-closed. Revalidate on the campaign's locked CUDA stack and verify deterministic evaluator parity from the selected Q3 parent. No Q4A research run exists. |
| Q4B | Hybrid categorical/Gaussian actor and PPO policy wiring exist; focused test source covers sampling, log-probability, entropy, gradients, and save/load. A bounded MI210 hybrid PPO update and save/load check is recorded in `docs/operations/triton_environment.md`. | It is opt-in and requires the shared actor architecture, a declared overlay `ActionSchema`, and an active exit coordinate when `exit` is present. Validate every category center against the environment decoder and campaign-locked trainer/evaluator/checkpoint integration. Registry arm remains fail-closed; no Q4B research run exists. |
| Q5 | Implemented in the training callback as primal-dual scalarized-return PPO. Each transition's cost is the maximum of the environment's positive normalized daily/max-loss/trailing risk-ratio increments; episode cost is their sum over a full chronological training split. PPO receives `reward - lambda * cost`; the nonnegative multiplier updates from completed episodes against a configured episode cost budget. The multiplier/config version is stored on the SB3 model for checkpoint persistence. Hard account guardrails remain unchanged. Tests cover reward shaping, dual update, restore/config compatibility, full-split PPO-only gating, and SB3 checkpoint round-trip. Current-tree full Ruff format/lint, Mypy, and all 1,603 tests passed on MI210; an earlier H200 suite passed 1,594 tests before the latest worktree changes. | Freeze and record the Q4 parent/checkpoint, validate persisted Q5 state through the campaign's actual checkpoint-load path, and run deterministic full train/test evaluation with economic and survival metrics separated. The default budget is a research starting point, not a calibrated or guaranteed risk bound. Q5 remains unlaunched and blocked on Q4 parent freeze. |
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

## Verification boundary

The current dirty source tree was validated on MI210 job `21012389`, step 3, at
HEAD `599741fe710fa8b342ef3797448132413c07e4db`. The source diff for the
validated implementation files had SHA-256
`10367a0c445ee6005ac8923464b85b3a810d0dc7cd402d7ce864359cb0ab445b`. Full-tree
Ruff format and lint passed; full-tree Mypy passed for 458 source files; a real
ROCm 6.3.1 / PyTorch 2.7.1+rocm6.3 MI210 matrix operation passed; and all 1,603
tests passed with 38 warnings. The full log is
`outputs/gpu_ci_gates/q0_q7_current_tree_20261017_job21012389/current_tree_validation.log`.

Earlier on 2026-10-17, H200 job `21004765` on `gpu59` (H200 x1, 48 CPUs,
512 GiB) passed CUDA matrix smoke, full-tree Ruff format/lint, full-tree Mypy,
and 1,594 tests (`gpu_ci_final3.log`). The initial MI210 validation in job
`21012389` also passed 1,594 tests (`mi210_full_validation.log`); the current
tree result above adds the updated source and 1,603-test result. No training
run, checkpoint mutation, cancellation, Q-stage campaign, or stage promotion
occurred. Slurm's one-minute allocation wait rule and A100-or-better/MI210
fallback are already present in `.agents/rules/triton-slurm.md`; do not create
a duplicate rule.
