# Thesis Execution Plan

**Status:** ACTIVE. The 18-variant repair screen is terminal. The frozen RQ1 pair is in a fresh seed-50 viability run; the five-seed comparison has not started. P0–P5 are complete; P6–P11 remain.
**Revision:** 2026-10-21 Helsinki — focused on the active P6 run and remaining thesis gates
**Objective:** Produce scientifically valid empirical training results, evaluate and iterate, then submit a defensible thesis.
**Deadlines:** Supervisor PDF by 2026-11-09; faculty submission by 2026-11-23.

## Operating rule

**Minimum viable scientific validity → training → evaluation → iteration → final results.** Keep the operational queue limited to demonstrated failures that could invalidate training, comparison, recovery, or submission. Do not delay training for speculative audits or infrastructure.

Use the supplied data through 2026-10-16. Train on 2024-12-30–2025-12-31; report on 2026-01-01–2026-10-16. Keep all bars, including short-session bars. The early-close schedule is known before each date starts and must be available from that date's first bar. Training and report inputs/caches remain split. M1 is the policy timeline; supplied M3–D1 bars and bid/ask ticks remain available for their configured features and execution. The 2026 interval has already been inspected, so its results are exposed exploratory evidence, not a fresh holdout.

## Current state and evidence

- **P0–P3 complete:** chronological split, date-known session schedule, retained timeframe/tick inputs, and deterministic checkpoint recovery are recorded in the relevant run manifests and `outputs/gpu_campaigns/replacement18_seed50_train2025_test2026_current_mtf_20261019/p3_smokes/job_21040072/p3_recovery_evidence.json`. The smoke also failed its training risk-survival gate; recovery evidence is not strategy viability evidence.
- **P4 complete:** the 18-variant corrective screen ended with 15 successes and 3 unchanged training risk-gate failures. One reported success only re-evaluated an inherited 20M checkpoint with no new updates. The run used warm starts and an explicit fingerprint override; preserve its canonical identity and classify its outcomes as exploratory. Do not retry failures or weaken the gate. Terminal summary: `outputs/gpu_campaigns/replacement18_iter1_seed50_train2025_test2026_20261020_reorg1/reports/campaign_results_summary_terminal_20261021.md`.
- **P5 complete:** retain the predeclared pair: control `ladder_a4b_entry_fsm`, treatment `ladder_a4a_entry_fsm_hidden`; the saved configurations differ only in `env.entry_state_observation`. The training-only freeze and identities are in `docs/thesis/evidence/p6_selected_pair_freeze_20261019.json`. Earlier matched training reports had zero trades in both arms. A later warm-start continuation had 6/7 trades but zero treatment reward, so it did not pass P6.
- **P6 active:** fresh scratch campaign `p6_entry_fsm_viability_seed50_train2025_20261021_scratch1`, fingerprint `3ca5c8e0cd3df51b29a5cdcb8f3d1c0f602a6c6721bf5aed3f6986a851b769b7`, seed 50, 20M requested steps per arm, 2025 training data. Allocation `21103722` runs on `gpuarm1` and `gpuarm2` with one GH200, 48 CPUs, and 512 GiB per node. At 08:02 EEST both workers had reached 393,216/20M steps at about 1,758–1,769 FPS; the indicative remaining training estimate was about 3 hours 5 minutes, excluding evaluation. The `p6-pair-scratch-20261021` supervisor is active. This is a training-only viability pilot; it does not establish robustness to the configured 5-point slippage because strict tick execution returns raw tick quotes. Generated status: `outputs/gpu_campaigns/p6_entry_fsm_viability_seed50_train2025_20261021_scratch1/reports/campaign_status_20261021_0802EEST.md`.
- **P7–P11:** no eligible five-seed paired results or final thesis claim exist yet. The manuscript is a working methods/results draft at `docs/thesis/manuscript_draft.md`; do not fill in claims from interim worker logs or report-only 2026 outcomes.

## Remaining work: blockers, deferred work, and removals

| Classification | Concrete failure or reason | Minimum acceptance |
|---|---|---|
| **BLOCKING — P6 viability, current** | The earlier warm-start treatment had zero reconciled training reward; it cannot establish that the frozen pair learns under the repaired source. The scratch run is active and has no terminal evidence yet. | Both fresh arms complete their declared 20M training budget on matching source, data, and runtime identities; pass the unchanged risk-survival gate; show lifecycle progression and nonzero reconciled training reward in each arm. Decide from 2025 training evidence only. Preserve zero-trade/failure outcomes. |
| **BLOCKING — P7 paired RQ1 evidence, after P6** | No valid five-seed paired estimate exists. The earlier seed-101/102 attempts came from defective source and are ineligible. Preserve paused campaign `rq1_entry_fsm_pair_seed101_105_train2025_test2026_20261019` and fingerprint `5f3855a472e890be0f37556372bdbdd538d97b21e435ad900f9803ac53bd4dce`; do not resume or reuse its checkpoints. The 2026 interval is exposed. | If P6 passes, start both arms from scratch for matched seeds 101–105, 20M steps per arm, under a new fingerprint and frozen configs. Report each arm's `100 × total_pnl / initial_balance`, each paired difference (hidden treatment minus visible control), the mean and two-sided 95% Student-t interval (df=4). Retain zero-trade pairs and incomplete seeds. Label the estimate exploratory and conditional on this fixed exposed interval. Do not select or tune using 2026 results. |
| **BLOCKING — selected-pair cost stress, after valid checkpoints** | The saved config sets `costs.slippage_points=5.0`, `use_tick_execution=true`, and 30 ms randomized latency. The strict tick path returns raw tick quotes; the configured slippage is not applied to those fills (`simulation/environment/tick_quote_runtime.py`; only fallback level fills use it in `fill_quote_runtime.py`). Therefore changing the YAML value alone cannot establish cost robustness. This does not block P6's matched learning-viability decision, but it limits cost claims. | After valid checkpoints, use an evaluation-only adverse-price override after selecting the tick, with effective 5.0-point baseline and 10.0-point stress fills. Preserve raw tick selection, timestamp, latency draw, execution seed, checkpoint, and data. Focused long/short fill checks and repeated paired replay must reproduce exactly; record the original/effective config and hashes and disclose that P6/P7 training used raw tick quotes without this extra slippage. |
| **BLOCKING — P8 independent review** | Without a reviewer checking the selected claim against the exact evidence, interpretation or validity defects may survive into submission. | Independent reviewer receives the frozen pair protocol, complete selected-pair reports, limitations, and manuscript; record each issue and its resolution or explicit limitation. |
| **BLOCKING — P9 result freeze** | A manuscript claim cannot be reconstructed if its source, data, checkpoint, or report identity is missing or changes after review. | Freeze hashes and exact identities for the results actually reported, their analysis, uncertainty, and claim limits. Reopen only for a demonstrated validity defect. |
| **BLOCKING — P10/P11 submission** | Missing or unreadable final files would miss the supervisor and faculty deliverables. | Produce and inspect the supervisor PDF by 2026-11-09; incorporate required corrections; retain hashes of the exact final manuscript and required artifacts submitted by 2026-11-23. |
| **DEFERRED** | Parity checks or additional robustness beyond the selected pair, broad uncertainty methods, extra seeds, additional architectures, and manuscript polishing do not unblock the active P6 training. | Consider only after P6/P7 and only if needed to support a specific thesis claim. Do not expand the experiment set by default. |
| **REMOVE FROM ACTIVE SCOPE** | Repeated FTMO/data-provenance audits, whole-screen backend parity, duplicate smoke/test runs, speculative cache/runtime infrastructure, and polishing reports for every variant do not materially improve the selected-pair evidence. | Do not restart these tasks unless an observed failure in the selected result path makes one necessary. Preserve existing user data and evidence. |

## P0–P11 dependency order

| Stage | State and exit evidence |
|---|---|
| **P0 — Data split** | Complete. Train 2024-12-30–2025-12-31; report 2026-01-01–2026-10-16. Retain all supplied bars and split caches. |
| **P1 — Inputs and features** | Complete for the active run. Date-known short-session schedule; M1 policy timeline; supplied M3–D1 context; bid/ask ticks retained. |
| **P2 — Execution** | Complete for the active run. Seeded 0–30 ms arrival delay and first eligible tick at or after order arrival; required tick input fails closed. |
| **P3 — Recovery** | Complete for checkpoint restore and deterministic replay. The smoke's risk-gate failure remains visible and is not a viability pass. |
| **P4 — 18-variant repair screen** | Complete and terminal. Preserve all 18 outcomes, three risk-gate failures, no-update re-evaluation, checkpoint lineage, canonical fingerprint, and exploratory status. |
| **P5 — Candidate freeze** | Complete. Keep the predeclared pair and its config identities; no exposed 2026 data informed selection. |
| **P6 — Pair viability** | Active. Complete the fresh seed-50 scratch pilot and apply the training-only gate above. Do not start P7 until it passes. |
| **P7 — Paired results and selected-pair stress** | Pending P6. Run the five matched seeds, analyze the frozen endpoint, and complete only the specified selected-pair cost stress. All 2026 results remain exposed/report-only. |
| **P8 — Independent review** | After P7 evidence is complete; resolve or disclose review findings. |
| **P9 — Freeze results** | After review; freeze exact evidence, hashes, analysis, and claim limits. |
| **P10 — Supervisor PDF** | By 2026-11-09; inspect the rendered PDF and incorporate required feedback. |
| **P11 — Faculty submission** | By 2026-11-23; retain exact submitted artifacts and hashes. The user handles administrative reporting. |

## Execution and preservation rules

- Work only in `/scratch/work/nguyenl37/Aalto_MS_Thesis`, on the existing `main` checkout. Do not create branches or worktrees.
- Preserve dirty user work, raw data, fingerprints, manifests, checkpoints, supervisors, allocations, and existing worktrees. Never run `scancel` or create a duplicate campaign.
- Before a new Slurm request, follow `.agents/rules/triton-slurm.md`: inspect per-node GPU/CPU/RAM, use a named persistent login tmux, request the approved shape with `--immediate=60`, and preserve pending requests. Keep the two-GH200 aggregate cap.
- For long-running work, estimate the next useful check from observed throughput and the allocation limit; continue independent thesis work while it runs. Use `experiments/campaigns/status.py` for meaningful campaign status updates and include its generated report.
- A fingerprint mismatch may use the explicit reason-based resume override only when the controller's continuity checks pass. Preserve the canonical fingerprint, record the effective identity and reason, and mark all resumed evidence exploratory. Never edit a fingerprint to force a match.
- Run only scoped tests needed for a demonstrated code change; use the required full-tree Ruff and Mypy gate before any code commit. Do not rerun the full test suite without a concrete need.
