# Zero-trade prevention validation — 2026-10-23

## Decision

The fail-closed activity protections and the selected-pair mechanical preflight pass. Both 5M seed-50 checkpoints complete a deterministic replay over the full 2025 training calendar with 33 closed trades and no survival breach. This proves minimum activity and replay viability for these two exploratory runs; it does not prove profitability or adaptive policy quality.

**NO-GO FOR 20M production training at this time.** The referenced repository-local `ZERO_TRADE_PREVENTION_AGENT_PLAN.md` is absent, so its formal GO criteria cannot be audited. The supplied plan text is implemented and tested, but the 5M evidence is one seed per arm and the deterministic action outputs are nearly constant. Do not treat this report as authorization to launch a 20M run.

## Files changed

The zero-trade protection work is on `main`, pushed through commit `d7515f7` (`Persist deterministic viability diagnostics`), following `80af33a` (`Reject inactive deterministic training runs`), `d170e46` (`Replay full training split for viability gates`), and `b58da5f` (`Add selected-pair scripted reachability preflight`). The changes cover:

- `training/callbacks/activity.py`, `training/runner/seed_runtime.py`, and `training/runner/reporting_runtime.py`: separate survival and deterministic activity gates, report inactive runs as `FAIL_INACTIVE`, persist diagnoses, and fail closed after writing the training log.
- `simulation/environment/action_intent_runtime.py` and `simulation/environment/diagnostics_runtime.py`: honor a valid stored FSM candidate direction through an active manipulation state.
- `learning/rl/callbacks.py`: deterministic full-training-calendar viability checks, action-coordinate summaries, entry-funnel counters, and exclusion of inactive checkpoints from best-checkpoint selection.
- `scripts/research/audits/audit_scripted_training_reachability.py` and `docs/thesis/evidence/scripted_training_reachability_20261023.json`: resolved-config, real-data reachability preflight.
- Regression coverage in `tests/test_train/test_training_activity_gate.py`, `tests/test_train/test_checkpoint_eval_determinism.py`, `tests/test_train/test_train_rl_decomposition.py`, and `tests/test_envs/test_entry_state_machine.py`.

## Tests and checks

All checks below ran on GPU nodes. The current focused gate ran on GH200 node `gpuarm2`, allocation `21173947`.

| Command | Result |
|---|---|
| `ruff format --check .` | Pass; 742 files formatted |
| `ruff check .` | Pass |
| Full-tree Mypy using the documented node-local staging procedure | Pass; 742 source files, no issues |
| `python -m pytest tests/test_train/test_checkpoint_eval_determinism.py tests/test_train/test_train_rl_decomposition.py tests/test_train/test_training_activity_gate.py tests/test_train/test_training_smoke.py tests/test_envs/test_entry_state_machine.py -q` | 58 passed, 1 unrelated Transformer nested-tensor warning |

The broader full pytest suite was not rerun. The five focused files cover the activity gate, fail-closed finalization, deterministic viability callback, checkpoint eligibility, and FSM direction/manipulation regressions.

## Gate status

| Requirement | Status | Evidence |
|---|---|---|
| Activity gate | **PASS** | 0 trades is inactive; 1–4 is `ACTIVE_SPARSE`; 5+ is `ACTIVE`. A survival pass plus zero deterministic train trades writes `FAIL_INACTIVE`, `training_admissible=false`, and raises `TRAIN_ACTIVITY_FAILED`. |
| Survival/activity/performance separation | **PASS** | Survival and activity have separate fields and failure paths; activity does not imply profitability. |
| FSM and manipulation fix | **PASS** | Regression test exercises long and short candidates both with and without active manipulation. |
| Deterministic viability monitor | **PASS** | Runs on the full chronological training split at 5M threshold crossings and records trade count, survival, action coordinates, and entry funnel. |
| Zero-trade diagnosis | **PASS for available deterministic diagnostics** | Tests cover missing candidates, failed arming, policy inactivity, execution refusal, risk floor, stop geometry, and target geometry. No real zero-trade final replay occurred in these pilots, so no real zero-trade root cause is claimed. |
| Best-checkpoint activity filter | **PASS** | Checkpoint callback refuses zero-trade, breached, or non-finite evaluations; tests cover deterministic action use and inactive checkpoint rejection. |
| Real-data reachability | **PASS** | Both selected variants produced 48 scripted opens and closes each, zero breaches, on 351,432 training bars from 2024-12-30 through 2025-12-31. No test split was evaluated. |
| Effective-config preflight | **PASS** | The audit resolved both production variants, verified that their only difference is `env.entry_state_observation`, and recorded per-variant config hashes. |

Reachability evidence SHA-256: `047250f5fb712aa1fa2af9e64d47660932eb22c76cf40ab7e6ad558bbb34715f`.

## Staged training evidence

| Stage | Variant / seed | Result | Interpretation |
|---|---|---|---|
| 100k smoke | `ladder_a4a_entry_fsm_hidden`, seed 50 | 106,496 steps; active checkpoint available with 1 trade in its checkpoint evaluation; process exit 0. | Completed on GH200 with 16 environments. Test evaluation was not called. The initial 64-environment attempt was interrupted during environment startup; the bounded retry completed. |
| 1M pilot | `ladder_a4b_entry_fsm`, seed 50 | 1,007,616 steps; final checkpoint saved; 1 trade in the 20k-bar training-prefix monitor. | Training-only pilot. This short prefix is not the full-year deterministic activity result. |
| 5M pilot | `ladder_a4b_entry_fsm`, seed 50 | 5,005,312 steps; full-year deterministic replay: 33 closed trades, survival valid, finite equity. | `ACTIVE` by the declared minimum; full-year replay took 554.425 seconds. |
| 5M pilot | `ladder_a4a_entry_fsm_hidden`, seed 50 | 5,005,312 steps; full-year deterministic replay: 33 closed trades, survival valid, finite equity. | `ACTIVE` by the declared minimum; full-year replay took 569.786 seconds. |
| 10M stage | — | Not run. | Not required by the supplied staged plan because both 5M full-year deterministic checks were active and survived. |

The complete 5M replay rows, including action-coordinate statistics and entry-funnel counters, are in the ignored run artifact `outputs/zero_trade_deterministic_replays_20261023.jsonl` (SHA-256 `f142ffd05a16ea03e02d824a8af8054f3c566334318f67c39b58190bf17c3265`). Both replays used the full 2025 training calendar only; neither called test evaluation.

During stochastic PPO training, the episode logs recorded 2,015 sampled trades across 48 episodes for `a4a` and 1,978 across 48 episodes for `a4b`. Four `a4a` and three `a4b` exploratory episodes recorded breaches. These exploratory episode counts and breaches are distinct from the final deterministic full-year replays, which each closed 33 trades with no breach.

**Action-collapse risk:** over 99,492 deterministic training bars, all action-coordinate standard deviations were at most `1.2e-6` for both 5M checkpoints. The models were active, but these pilots do not establish state-responsive or economically useful learned behavior. The short 20k-bar prefix monitors also recorded no trades at the 5M checkpoints; those prefix results are not substitutes for the complete training-calendar replays.

## Remaining risks and 20M decision

- The repository-local `ZERO_TRADE_PREVENTION_AGENT_PLAN.md` named by the task is missing. Its unprovided formal GO criteria cannot be verified from this checkout.
- The two 5M pilots use one seed per arm. They are viability diagnostics, not statistical comparison or confirmatory performance evidence.
- Deterministic action outputs are nearly constant despite nonzero activity; investigate this as a policy-learning question without changing rewards, weakening risk limits, or selecting on 2026.
- The pilots directly called the seed runner's training path. The fail-closed final reporting path is unit-tested, but these pilot runs did not execute a production 20M finalization.
- No 2026 test/report-only results are used here.

**Final decision: NO-GO FOR 20M production training** until the repository's formal GO criteria are restored and reviewed. The implementation and 5M minimum-activity checks pass; this decision reflects the missing formal gate and unresolved policy-learning evidence, not a zero-trade failure in the selected pair.
