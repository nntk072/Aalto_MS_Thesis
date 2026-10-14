# 18-Variant Training Campaign — Readiness Investigation and Implementation Plan

**Date:** 2026-10-10 (Europe/Helsinki)<br>
**Triton repo:** /scratch/work/nguyenl37/Aalto_MS_Thesis<br>
**Observed HEAD:** b00b5e85622ec1410c8a11cc063cd52c87d3976d (working tree clean before creating this plan)<br>
**Scope:** Implementation authorized by follow-up. Repository configuration, campaign provenance, artifact validation and evaluation metadata are being implemented. No production training, Slurm allocation, benchmark, or commit is authorized by this document.

## 1. Executive summary and decision

**Readiness: NOT READY for production or confirmatory claims.** The researcher specified `ladder_a0_flat` as a pure legacy reference (`strategy_actions=false`, legacy `Discrete(20)`, no PO3 manipulation guard). The previous 100k pilot's A0 artifact used `strategy_actions=true`; it is not evidence for that reference, and A0 is not a one-factor control for A1. The authorized exploratory campaign is now running as Slurm job `20954895`: seed 42, 20,000,000 requested steps per variant, shared-bundle backend, `n_envs=64`, and the chronological 2025 training / 2026 report-only test split. Its test interval was previously inspected, so results are selection-acknowledged and exploratory. The post-correction preflight passed fixed-trace frames/bundle parity for all 18 variants; see the campaign parity report. The old 100k pilot failures remain historical evidence and do not determine the outcome of this longer run. The active workers use source commit `937bae5`; later metric-policy changes in `16b13ae` do not affect them and prevent direct resume from the updated checkout. Current-source feature-prefix checks passed nine cutoffs across three profiles. Sustained full-run resource sizing, independent research review, and a viable strategy remain open. The researcher waived new post-2026-07-03 data for exploratory work only; confirmatory claims still require an untouched holdout.

**Live status refresh — 2026-10-14 13:09 UTC:** Slurm job `20954895` is RUNNING on two GH200 nodes, with 18 active unique lineages reconciled across the primary manifest (16 active, two cache-start failures) and the recovery manifest (two active). Progress is 2.949M–4.620M/20M logged steps; no early-abort marker or final train/test report is present. At 3.32 hours, allocation cgroup RAM averaged 141.2/141.8 GiB and peaked at 158.7/162.6 GiB; CPU averaged 3.82/4.89 cores; GPU utilization averaged 56.5%/55.6%; VRAM peaked at 50,499/58,175 MiB; OOM counters were zero. See `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/live_campaign_snapshot_20261014_1309.json` (SHA-256 `af80c4cf4d7be47d9adb1a9caad263970559a619c9f51bf833e97f0e11075e07`) for worker-level logs and raw resource CSV hashes. A recovery-controller log has validation-error text despite both recovery lineages running; preserve that command-path discrepancy for audit. These are interim exploratory observations, not evidence of a successful policy. Readiness stays **NOT READY**.

**Checkpoint recovery refresh — 2026-10-14 13:28 UTC:** the latest selected checkpoint for all 18 lineages passed CPU deserialization and saved-timestep validation (18/18; 3M–4M-step checkpoints). Audit: `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_live_20261014_1316.json` (SHA-256 `2ecff25d092504db675dcceb0ab38b08d996a81c31a63be89af471ca989fb9f7`). Job `20954895` remains RUNNING at 4:03/6:00, with logged worker progress 3.146M–5.112M steps and the migration watcher waiting. CPU loading does not establish GH200 resume or final model validity. Readiness stays **NOT READY**.

**Live worker refresh — 2026-10-14 13:32 UTC:** Slurm job `20954895` is RUNNING at 4:06:32/6:00. All 18 unique worker logs are advancing at 3.244M–5.210M PPO timesteps; seven latest 20,000-bar training-prefix monitors report zero trades. No `EARLY_ABORT` marker or final train/test evaluation artifact was found. These monitors are interim diagnostics, not final outcomes. Hashed log-level evidence is in `docs/thesis/live_campaign_snapshot_2026-10-14_1332.json` (SHA-256 `05eb275738e408b2b1c47d84c54eba2e7ded9d8eb3d94be1e6bc913f0a810c91`). Readiness stays **NOT READY**.

**Live checkpoint and worker refresh — 2026-10-14 13:48 UTC:** latest resumable checkpoints for all 18 unique lineages passed provenance/hash checks and CPU deserialization (18/18; selected checkpoints at 3M–5M steps). The audit is `docs/thesis/checkpoint_load_cpu_18_seed42_live_20261014_1338.json` (SHA-256 `75581ab0b06107c232e458479673d57134dfcac167884a2f080cfdd31e41f1e3`). Slurm job `20954895` remained RUNNING at 4:23:13/6:00, with live worker progress at 3.441M–5.702M steps. Six latest 20,000-bar training-prefix monitors show zero trades; no `EARLY_ABORT` marker or final train/test evaluation was found. Snapshot: `docs/thesis/live_campaign_snapshot_2026-10-14_1348.json` (SHA-256 `efdf664e4d32320d601fdc41f3ac581d19fcdb1d947b9fa01e40b9428aac17d9`). These are interim exploratory observations; readiness stays **NOT READY**.

**Four-hour resource refresh — 2026-10-14 13:52 UTC:** over 4.02 hours, process-tree CPU averaged 3.82/4.89 cores on the 48-core nodes; cgroup RAM averaged 144.4/145.5 GiB and peaked at 166.5/170.0 GiB against 512 GiB per node. GPU utilization averaged 57.0%/55.9%, VRAM peaked at 51.2/60.0 GiB, and sampled OOM counters were zero. See the raw resource CSV hashes in `docs/thesis/live_campaign_snapshot_2026-10-14_1352.json` (SHA-256 `fc11afa4ac1779fb1179bbac452e369fa3fbfce262a536c010483d058082f209`). This remains bounded interim resource evidence; final evaluations and full-run sizing are still open.

The independent-review gate is prepared but not satisfied. See the
[`independent research review checklist`](../thesis/independent_research_review_checklist_2026-10-14.md);
it requires a qualified external reviewer and an explicit disposition.

**Status addendum — 2026-10-13:** the configured 2026 test period was previously
inspected and, under explicit researcher instruction, is now exposed development
data for exploratory tuning with selection acknowledgment. It is not a final
holdout. A bounded A0/A1 seed-42, 100,000-step-per-variant comparison is
prepared for development use. Fresh post-2026-07-03 data is required only for a
future confirmatory final-holdout claim; the researcher waives it for this
exploratory continuation. The 18 production runs remain not ready.

**Status addendum — 2026-10-13 (researcher data-scope clarification and source review):**
the researcher explicitly authorized continuing with the currently available
data and said post-2026-07-03 data is not required for this work. This waives
fresh-data acquisition as a prerequisite for the requested **exploratory**
continuation; it does not restore a pristine holdout. Keep the 2026 interval
`selection_acknowledged`, and do not describe its results as confirmatory. The
training/execution/reward/termination/checkpoint paths have now had a documented
source-level review in `docs/thesis/research_path_review_2026-10-13.md`. The
review records the stochastic one-episode `ppo_best` diagnostic and the risk
that a human could select after viewing exposed results; independent research
sign-off and any selection record remain open.

**Status addendum — 2026-10-13 (real-market parity and recovery):** all 18
registered variants passed a paired frames/bundle real-market replay. Each
variant was compared over the full 49,925-step calendar with an all-zero action
vector and a 10,000-step active-action trace; observations, rewards, step
metadata, trade/account/equity state, and final counters matched exactly
(maximum scalar delta 0.0). The zero vector can still cause trades in
context-driven strategies, so it is not described as a hold. Provenance,
including raw M1 and tick-cache SHA-256 values, is recorded in
`outputs/gpu_campaigns/real_market_frames_bundle_parity_20261013/reports/frames_bundle_parity.json`.
The bundle worker now forwards `strict_research_eligibility` and `allow_simplex`
from the effective config; the focused regression and real replay pass.
Controller failures now retain artifact hashes and failure reasons per attempt;
successful admission also requires the attempt provenance to match the campaign
fingerprint, run key, seed, budget and attempt. The parity and failure
path gates in this plan are closed for the tested source/data scope. The
unrecoverable legacy A2/A3 archives are formally excluded from future candidate
ranking and confirmatory claims by
`docs/thesis/legacy_a2_a3_checkpoint_exclusion_2026-10-13.json`; originals
remain untouched. This does
not establish exhaustive feature causality or real-data train/test leakage
safety. Targeted causal-feature, split, train-only fit-scope, walk-forward,
normalization, resolver, shared-bundle, and eligibility tests pass (49 tests,
2 warnings); the full suite passes (1,348 tests, 38 warnings). Production readiness remains **NOT READY** pending real-data
causality/leakage evidence, sustained resource sizing, fresh final-holdout
performance and separate launch authorization.

**Status addendum — 2026-10-13 (real-market feature causality):** the grouped
prefix-invariance audit passed all 6 profile/cutoff checks across the 18
variants (418 baseline features; 486 PO3/distribution features). The exact
train/test seam audit passed 4 additional checks at the last training bar
(2025-12-31 23:49) and first test bar (2026-01-02 01:05). Every compared value
matched (max absolute delta 0.0; no changed features). Reports record input and
source hashes in `outputs/gpu_campaigns/real_market_feature_causality_20261013/`.
This closes real-M1 feature-prefix invariance for the tested cutoffs; execution,
reward/termination and model-selection paths still require separate review.

**Status addendum — 2026-10-13 (sustained one-worker resource profile and final local gates):** one A0 seed-42 run reached 2,007,040 completed steps for the 2,000,000-step request, and its checkpoint loads on CPU. It is a resource profile on exposed development data, not a candidate-selection result. Training failed the mandatory equity/survival gate (−10.23% end equity, negative equity slope, 10.82% peak trailing drawdown, and max-loss stop); test evaluation remained inactive with zero trades. The controller marked the campaign failed. Across 93 resource samples, process-tree PSS peaked at 45.4 GiB, GPU memory at 5,439 MiB and sampled utilization at 28%. Cgroup memory telemetry was unavailable; the process-level I/O counters from this run are unusable because the sampler process had loaded the pre-fix parser. Slurm step counters recorded approximate deltas of 146.8 GB read and 137.9 GB written, scoped to the whole allocation step. Full details and limits are in `outputs/gpu_campaigns/resource_profile_a0_seed42_2m_exposed_20261013/reports/resource_profile_summary.json`. The local Ruff, format and all-source mypy gates pass. The final full-suite rerun is recorded separately after completion. This evidence is insufficient to size production concurrency or establish a passing strategy. The A0/A1 100,000-step exposed comparison and this single A0 long run both fail training equity acceptance. Readiness remains **NOT READY** pending a passing strategy, fresh post-2026-07-03 holdout data, review of execution/reward/termination and model selection, production resource sizing, and separate launch authorization.

**Final local verification — 2026-10-13:** `ruff check .`, `ruff format --check .`, `mypy .` (389 source files), `git diff --check`, and resource-summary JSON parsing pass. Full pytest passes **1,351 tests, 38 warnings** in 487.03 seconds. The complete log is `outputs/gpu_campaigns/readiness_verification_20261013/logs/full_pytest_final.log` (exit code 0).

**Status addendum — 2026-10-13 (resource telemetry correction):** the sampler
and campaign controller now resolve cgroup v1/v2 memory files using the
current process's nested cgroup path rather than assuming the mount root is the
job scope. The sampler also records the cgroup memory limit. Three focused
tests cover nested v2, v1, and path-escape cases. This fixes telemetry lookup
and passes local checks; it does not itself establish Slurm memory peaks,
process I/O, sustained throughput, or safe production concurrency. No new GPU
profile or 18-run campaign was launched in this continuation. Existing A0/A1
pilot failures remain hard acceptance failures; do not weaken them. The campaign
therefore remains **NOT READY**.

**Evidence labels:** VERIFIED means source/current remote state or inspected report. HISTORICAL means existing documentation or earlier validation. DERIVED means arithmetic from recorded inputs. OPEN/HYPOTHESIS means an experiment or audit is still necessary. Do not treat passing fixture/unit tests as complete real-data trajectory parity. Do not weaken risk, reward, execution or acceptance criteria to force readiness.

## 2. Authoritative 18-variant inventory

**Registry:** scripts/variant_list.txt, config/experiments.yaml and quant_rl/train/variant_resolver.py. The knob_* and ablation_* experiments in experiments.yaml are separate and excluded. All core variants select PPO, TCN and use_vae=false. `ladder_a0_flat` is now the pure legacy baseline (`strategy_actions=false`); the other 17 use `strategy_actions=true`. The resolver therefore selects the default feature profile and legacy `Discrete(20)` action path for A0, and the full PO3 feature profile for the other baseline-family rungs. PPO default values include n_steps=2048, batch_size=512, n_epochs=15, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95, clip_range=0.2. Check scaling before training; reported effective configuration must be post-merge and post-scaling.

The action widths and observation branches below were instantiated for all 18 registry variants on real cached market features and recorded with effective environment flags in `outputs/gpu_campaigns/dev_a0_a1_seed42_100k_exposed_2026/reports/action_space_audit.json`.

| # | Variant | Strategy, reward | Intended change | Effective direction / SL / TP / multi-TP / FSM | Instantiated width |
| ---: | --- | --- | --- | --- | ---: |
| 1 | ladder_a0_flat | baseline, dsr | Pure legacy A0 reference | legacy discrete / off / off / off / off | Discrete(20) |
| 2 | ladder_a1_sl | baseline, dsr | A1 SL | on / on / off / off / off | 7 |
| 3 | ladder_a2_tp | baseline, dsr | A2 TP mode | on / on / on / off / off | 8 |
| 4 | ladder_a3_multi_tp | baseline, dsr | A3 multi-TP | on / on / on / on / off | 12 |
| 5 | ladder_a4b_entry_fsm | baseline, dsr | A4 visible FSM | on / on / on / on / visible | 12 |
| 6 | ladder_a4a_entry_fsm_hidden | baseline, dsr | A4 hidden FSM | on / on / on / on / hidden | 12 |
| 7 | ladder_a0_flat_po3 | po3_ifvg, pnl | A0 PO3 | inherited off / off / off / off / off | 5 |
| 8 | ladder_a1_sl_po3 | po3_ifvg, pnl | A1 SL | inherited off / on / off / off / off | 6 |
| 9 | ladder_a2_tp_po3 | po3_ifvg, pnl | A2 TP mode | inherited off / on / on / off / off | 7 |
| 10 | ladder_a3_multi_tp_po3 | po3_ifvg, pnl | A3 multi-TP | inherited off / on / on / on / off | 11 |
| 11 | ladder_a4b_po3_ifvg | po3_ifvg, pnl | A4 visible FSM | inherited off / on / on / on / visible | 11 |
| 12 | ladder_a4a_entry_fsm_hidden_po3 | po3_ifvg, pnl | A4 hidden FSM | inherited off / on / on / on / hidden | 11 |
| 13 | ladder_a0_flat_dist | distribution, pnl | A0 distribution | inherited off / off / off / off / off | 5 |
| 14 | ladder_a1_sl_dist | distribution, pnl | A1 SL | inherited off / on / off / off / off | 6 |
| 15 | ladder_a2_tp_dist | distribution, pnl | A2 TP mode | inherited off / on / on / off / off | 7 |
| 16 | ladder_a3_multi_tp_dist | distribution, pnl | A3 multi-TP | inherited off / on / on / on / off | 11 |
| 17 | ladder_a4b_distribution | distribution, pnl | A4 visible FSM | inherited off / on / on / on / visible | 11 |
| 18 | ladder_a4a_entry_fsm_hidden_dist | distribution, pnl | A4 hidden FSM | inherited off / on / on / on / hidden | 11 |

The strategy families intentionally differ: baseline uses DSR reward, while PO3/distribution use PnL, have no direction head, and enable PD context/structure. The pure legacy A0 now also has a different action schema and feature profile from A1-A4, so A0 versus A1 is not an isolated SL ablation. The historical 18-variant action-space audit predates this choice and does not verify the new A0. Within A1-A4, A0 explicitly disables SL mode and A1 enables it, making that flag contrast explicit within each family. The prior instantiated audit confirms A4b exposes a 10-element account observation while A4a keeps the six-element account observation and masks the entry state; sequence shapes were 60x367 for baseline-family strategy-action runs, 60x411 for PO3, and 60x419 for distribution. Rerun the current A0 action/feature audit after this change.

**Shared source/config path:** quant_rl/config/default.yaml sets US100.cash M1 primary and US500.cash M1 secondary, broker timezone Etc/GMT-3, M1 source files dated 2024-12-30 through 2026-07-03, train_end=2025-12-31 and test_start=2026-01-01, feature window 60, normalization window 252 and M5/M15/H1 feature timeframes. Resolver enables session/liquidity/PO3/IFVG and strategy-state features; PO3/distribution add PD context and structure. It also sets env.open_manipulation_bars=0 and env.peak_trailing_dd_limit=0. Preserve intended risk/cost distinctions and snapshot their effective values.

**Training path:** scripts/train/train_one_variant.sh resolves the variant and launches quant_rl/train/train_rl.py; the latter merges base/strategy/CLI overrides, seeds Python/NumPy/Torch, creates masked feature caches and train/test splits, builds PPO/vector environments and trains. It writes periodic checkpoints, model/ppo_final.zip, per-run config.yaml and training_log.json; report data include requested/completed timesteps and OOS survival. The final test-split report uses deterministic actions and remains report-only. Frames is the frozen campaign backend. Bundle has exact real-market parity for the tested 18-variant calendar and active-action traces; exhaustive policy-state parity remains unproven. Bundle+VAE restriction is immaterial to the VAE-free core 18.


## 3. Existing validation and performance evidence

**VERIFIED artifacts:** outputs/gpu_campaigns/ram-validation-20261010/reports/bundle8.json, bundle16.json and matched_ppo_7day_8env_seed42_32768.json. Seven-day **environment-only** probes reduced incremental RAM from **2.98 to 1.05 GiB at eight envs** and **6.16 to 1.41 GiB at sixteen** (frames to bundle). At 16 envs measured process-tree PSS was **10,010 -> 6,590 MiB** and startup **26.61 -> 0.66 seconds**. These are not full PPO training RAM peaks.

The matched seven-day, eight-env, seed-42 **32,768-step PPO** report records: frames **41.0503 s, 798.24 steps/s, PSS 9,463.95 MiB**; bundle **31.8176 s, 1,029.87 steps/s, PSS 9,169.08 MiB**. Both recorded the same initial and final SHA-256 of concatenated model.policy.parameters() bytes, and matching sampled rollout rewards and PPO train logs. The hash does **not** include named full state_dict, persistent buffers, comprehensive trajectories, evaluations or all 18 variants. A report with identical parameter-byte hashes is supporting evidence, not proof of full scientific identity.

**VERIFIED concurrency reports:** concurrent_2x8_A/B.json report ~249.35+249.37=**498.71 aggregate steps/s**, worker tree PSS ~3,156/3,171 MiB. concurrent_2x16_A/B.json report ~530.25+536.75=**1,066.99 aggregate steps/s**, worker tree PSS ~3,920/3,934 MiB. These are short matched PPO profiles (32,768 steps, n_epochs=15, batch_size=512), not production-scale guarantees. Reported GPU peak allocations per worker were approximately 476 MiB in these runs, not aggregate physical VRAM peak.

**VERIFIED incomplete attempt:** `concurrent_4x8_local_v2/launcher_status.json` records all four workers exiting `-15` after 435.21 seconds, but logs A–D are empty and no worker report JSON exists. These retained artifacts establish no throughput and no root cause. The earlier draft's attribution to severe startup/filesystem latency is not verifiable from the files currently available; those causes remain hypotheses pending measured follow-up. **HISTORICAL:** 36 targeted tests, fixture parity and configuration regression reportedly passed; re-open their exact logs before making them acceptance evidence. Current-source real-data parity across all variants is now verified for the fixed traces in the completion record below; sustained full PPO resource peaks remain **OPEN**.

**VERIFIED Slurm snapshot (2026-10-10):** allocation **20879781**, RUNNING on gpuarm1 in gpu-grace-h200-141g, **64 CPUs, 400G RAM**, one NVIDIA GH200 144G observed previously, initial 12h walltime and ~11h42m remaining at inspection. This is a temporary allocation snapshot and is not a new workload authorization. Never assume original 20877341 or any allocation persists; requery before future use.

## 4. Severity / issue matrix

| Priority | Scope and source | Verified finding versus open question | Proposed action | Acceptance evidence |
| --- | --- | --- | --- | --- |
| **P0 scientific** | A0/A1; quant_rl/config/default.yaml:178; config/experiments.yaml; quant_rl/train/variant_resolver.py:40-174; quant_rl/envs/trading_env.py:839-850 | **Resolved and instantiated:** all three A0 entries explicitly set SL mode false; A1 remains true. All 18 variants constructed on real cached market features with intended action widths and observation branches. | Preserve the selected A0 meaning; rerun if registry, resolver, feature schema or environment action code changes. | `outputs/gpu_campaigns/dev_a0_a1_seed42_100k_exposed_2026/reports/action_space_audit.json`; full real-data backend trace report. |
| **P0 scientific** | feature/data split and cache; `quant_rl/train/train_rl.py`; `AGENTS.md` | **Implemented/source-reviewed:** feature-prefix checks and seam checks passed on the exposed real data; the training, execution, reward and selection paths have a documented source review. **Open:** independent research acceptance of the full execution/accounting/reward contract, human checkpoint/variant selection controls, and a pristine holdout for any confirmatory claim. The authorized data through 2026-07-03 is sufficient for exploratory work only. | Preserve the audit artifacts; keep the 2026 test interval acknowledged/exposed; require independent review and a new untouched split before confirmatory claims. | Prefix/causality reports and `docs/thesis/research_path_review_2026-10-13.md`; these do not constitute independent academic sign-off or a pristine holdout. |
| **P0 operational** | scripts/matrix/campaign_controller.py | **Implemented and exercised:** success requires matching attempt provenance, requested/completed budget, required metrics, a valid ZIP and a loadable PPO/SAC checkpoint. Resume revalidates successful artifacts. Failed attempts retain artifact hashes/reasons; retries get separate output trees. | Keep failed outputs and hashes immutable; revalidate successful artifacts at resume. | `tests/test_campaign_artifacts.py`: campaign and identity fingerprint mismatch, failed-worker hashes and successful attempt-2 resume; 13 tests pass. Full suite before this verifier refinement: 1,348 passed. |
| **P1 reproducibility** | config/experiments.yaml defaults, default.yaml and campaign_controller.py | **Resolved for this campaign:** the frozen spec selects one seed (42) and 20M requested steps; legacy registry/controller defaults remain historical. | Use `config/campaigns/18_variant_seed42_20m.yaml`; keep the spec authoritative and reject conflicting CLI identity overrides. | Dry-run exactly 18 unique keys; run manifests include effective config and source/data hashes. |
| **P1 reproducibility** | controller fingerprint/resume; train launcher | **Implemented and exercised:** fingerprints include tracked/untracked runtime source, campaign inputs, git diff and raw M1/tick checksums; each retry writes to its own output tree and appends logs. | Preserve attempt outputs and reject fingerprint changes on resume. | Failed-attempt hash plus successful attempt-2 resume integration test passes; controller mismatch guard remains enforced. |
| **P1 scientific** | TradingEnv action/SL/TP/FSM/reward/risk; shared bundle; entry gate | **Verified for bounded traces:** current-source parity passes all 18 variants over a 49,925-step exposed test-calendar trace and 10,000-step active-action trace; all 36 traces have maximum scalar delta 0.0. This does not prove exhaustive policy-state or every execution/label path. | Preserve the report and rerun if simulator, data, feature, resolver or strategy source changes. Complete separate causality and leakage reviews. | `outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.json`, SHA-256 `372e565c913aacd0c564e19f03503d749ee4d2b1a3988f8a9aac289ab3481044`. |
| **P1 scientific** | baseline versus strategy families; `quant_rl/train/train_rl.py` evaluation calls | **Current evidence contradicts acceptance:** the 18-worker 100k resource probe has 18/18 train-equity gate failures. On the previously inspected report-only test, 12/18 fail 130-day survival and the other six trade at most twice. The three-worker 20M run was interrupted before final evaluations. No candidate was selected. | Preserve every failure; do not rank or claim a matched causal strategy comparison. Any next study needs a frozen comparison design, explicit selection record and evaluation split roles. | Checksummed `outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md`; interrupted-run reconciliation under the `ladder3_strategy.../reports/` directory. |
| **P1 resources** | controller memory estimates, cgroup monitor, trainer launcher | **Measured bounded concurrency:** the 18-worker 100k probe ran with `n_envs=64`, six CPUs per worker and two GH200s on one node. Process-tree PSS averaged 218.9 GiB and peaked at 251.2 GiB; sampled GPU utilization averaged 66.4% per device and peaked at 97%/96%. Slurm step `TotalCPU` averages 0.35 core per worker step (5.8% of each six-CPU assignment). The old sampler CSV still lacks process-tree CPU samples and its cgroup fields do not establish job-wide scope. The current sampler writes CPU time and a cgroup-scope label; its end-to-end CSV contract is tested. **Open:** no sustained 20M phase-aware sample or reliable allocation-wide memory peak exists, and this does not size scratch I/O or the frozen frames/`n_envs=4` shape. | Retain these 100k values as a bounded baseline; any production shape requires sustained phase-aware telemetry with allocation-scoped memory and CPU evidence. | 95 sampler rows, 94 active; full derivation in `outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/slurm_cpu_accounting_20944844.md` / `.json` and raw `slurm_accounting_20944844.psv`; `tests/test_triton_resource_sampler.py` covers the current sampler CSV contract. |
| **P2 operations** | launcher, tmux, controller reporting | **Current state (2026-10-14):** allocation `20954895` is RUNNING on two GH200 nodes. Sixteen workers are in the primary manifest; two cache-recovery workers are running separately. Early step logs are partial, with no final evaluations or checkpoints at the latest inspection. The campaign source fingerprint is pinned to `937bae5`; direct resume from the current `main` checkout will fail fingerprint validation. | Preserve attempt outputs and checksums. Before the next allocation, resume under the exact pinned source or use an explicit, provenance-preserving checkpoint migration; never edit the old fingerprint to force resume. | Current Slurm state plus both campaign manifests, `source_snapshot.zip`, worker logs, and the live status addendum below. |
| **P2 evaluation** | checkpoint selection and risk-survival | **Open / no successful candidate:** all 18 current probe gates failed. The final deterministic train/test artifacts now distinguish in-sample and report-only roles for future runs; periodic dashboard EVAL is a progress monitor, not a final split. | Keep the independent train-equity/survival gate; require a selection record and separate untouched final holdout before confirmatory evaluation. | 18 saved `train_equity_gate.json` files and train/test metrics; all are failures, not passes. |

Confirmed configuration inheritance is an actionable scientific concern but the precise correction depends on the **intended A0 ablation definition**. Do not automatically change reward, loss limits, cost, action semantics or risk settings to manufacture a pass.

## 5. Recommended frozen configuration strategy

Use [`config/campaigns/18_variant_seed42_20m_purelegacy_shared_bundle_n64_20261014.yaml`](../../config/campaigns/18_variant_seed42_20m_purelegacy_shared_bundle_n64_20261014.yaml) for the currently authorized exploratory scope: exact 18-variant list, **one seed (42)**, **20,000,000 requested steps per variant**, shared-bundle data, one-year train and following report-only test windows, with `n_envs=64` and worker allocation settings supplied by the launcher. This is not the separate historical five-seed 90-run objective and does not establish confirmatory evidence.

Save every run's **fully merged effective OmegaConf**, resolver overrides and CLI, PPO values after device scaling, environment/account/action/feature flags, model and observation/action dimensions, hyperparameters, risk/cost/reward settings, strategy overlay checksum, feature schema and cache key, raw CSV and tick data SHA-256, source commit/dependency versions, GPU/CPU architecture, RNG source and worker seeds. Resolve dates using the configuration consumed by train_rl.py rather than assuming experiments.yaml's train_end/test_start propagate: those registry fields currently coincide with default.yaml but the resolver does not forward them, creating future-drift risk.

Keep legitimate variant-specific reward, direction, PD-context and FSM visibility contrasts documented. Enforce explicitly frozen settings within comparison groups. No VAE dependency for core 18. Keep output paths unique to campaign/variant/seed and never overwrite a completed run. Store attempt-specific immutable logs, checkpoint checksums, exit code, time, Slurm ID/host, effective config hash, training/evaluation metrics and failure reasons. Require model loadability and completed timestep budget for success, not merely the existence of model/ppo_final.zip.

## 6. One-seed scientific training/evaluation protocol

**Protocol clarification (2026-10-14):** profitability and activity are
reported metrics, not hard acceptance criteria. Complete one seed per variant
through the requested training budget, then evaluate the frozen policy on both
the full one-year training split and the following five-to-six-month test
split. The train survival gate rejects risk-limit breaches and peak trailing
drawdown above 10%; PnL, equity slope, and trade count remain reported metrics.
The currently inspected historical test split supports exploratory reporting
only. Do not reinterpret earlier probe gate results under this later protocol.

1. The researcher selected **20,000,000 steps** and seed **42** for all variants; the frozen campaign spec records both. One seed permits controlled comparison but cannot establish variance or significance.
2. Freeze source/data/splits, checksum raw market/tick inputs and features, train-only normalization, cache identity, exact timestamp alignment, warmup and purge/embargo. Verify independent locked OOS is reserved for final report; do not select checkpoints or variants on it.
3. Resolve and instantiate all 18 to prove exact observations, action heads, reward, risk/SL/TP/FSM flags and strategy differences. Assert A0/A1 intended relationship, decode staged TP levels and fractions, and include missing-entry-gate feature inspection.
4. Run (after implementation authorization) per-variant fixed-input frames/bundle differential real-market episodes and selected full trajectory comparisons. Match seed, starting balance, policy/actions and time/order; compare observations, reward components, trade state, stop/take profit, drawdowns and termination, with declared float tolerances. Diagnose first divergence.
5. On parity-verified backend, train with fully recorded effective PPO parameters, per-worker seeds, vectorization and initialization. Save complete final checkpoint and all persistent state metadata. Compare reruns **within same variant**; across variants different action spaces prevent assuming same initial parameter tensors.
6. Use separately frozen selection/validation data and deterministic final evaluation protocol to compare actions, trades, equity, returns, Sharpe definition, max DD, costs, win rates, risk survival, termination and SL/TP behavior. Preserve failures and evaluate locked OOS only according to final-report policy.
7. Publish an 18-row auditable output manifest and do not substitute similar aggregate performance for full behavioral equivalence.

### Live campaign status — 2026-10-14

The exploratory 18-variant, seed-42, 20M campaign has launched under Slurm
allocation `20954895` on gpuarm1/gpuarm2. Sixteen workers are in the primary
campaign and two stale-cache retries run in a separate recovery campaign.
The run is incomplete and remains exploratory because its report-only test
period was previously inspected. Its source identity is pinned to commit
`937bae529fdf5d6c17db6174880ef448320e2883`; metric-policy changes in
`16b13ae` do not alter its already-running workers and invalidate a direct
resume from the updated checkout. Preserve the fingerprint guard and source
provenance when continuing it. The current run does not close resource sizing,
independent research review, or pristine-holdout gaps; readiness remains
**NOT READY**.


## 7. Resources, conditional sizing and benchmark plan

**No trainer count or allocation is preselected.** Observe current squeue/scontrol, Slurm CPU allocation, cpuset and cgroup memory.max/current/peak/events, physical GPU inventory, active jobs, process-tree PSS, available scratch space/quota and disk throughput **at execution time**. Node-wide free memory is not the job's authorized RAM. Triton GH200 uses an aarch64 compatible environment; an x86 H100/H200 requires its matching environment. Respect AGENTS.md and .agents/rules/triton-slurm.md; project work runs on an allocated GPU compute node in persistent tmux. This planning task launches no jobs or Slurm steps.

After scientific parity/CI gates, benchmark **matched sustained PPO**, initially 1x8 and 1x16, then 2x8 and 2x16, for frames and approved bundle. Hold dataset, feature preprocessing, effective PPO parameters, model/variant, seed and step budget equal within comparisons, changing only backend or concurrency under test. Measure per-run and aggregate **completed PPO training steps/sec**, rollout steps/sec, PPO update time, total elapsed time, startup/cache preparation, checkpoint/evaluation overhead, CPU utilization and allocated CPU time, process-tree PSS and peaks, cgroup RAM peak/OOM events, GPU utilization and GPU VRAM peaks, filesystem throughput/latency, and walltime remainder. Distinguish measured peaks from samples, estimates or Slurm limits.

Investigate unsuccessful concurrent_4x8_local_v2 from logs: possible simultaneous cache creation, memmap page faults, filesystem metadata/open latency, scratch contention, environment start methods, and CPU saturation. Do not label any proposed cause proven until verified. Expand to three/four trainers only after reproducible health at two and demonstrated allocation headroom. Optimize **sustained aggregate completed PPO steps/hour** under memory/CPU/GPU/I/O limits; require restart/checkpoint headroom and confidence from repeated sustained runs.

**Derived ideal-duration sensitivity, not a prediction:** 18 x 100,000 = **1.8 million total steps**; 18 x 20,000,000 = **360 million**. At the short observed 2x16 aggregate rate **1,066.99 steps/sec**, the compute-only theoretical times are ~**0.47 h** and ~**93.7 h** respectively; at 2x8 rate **498.71 steps/sec**, 20M x 18 is ~**200.5 h**. These omit preprocess, long-horizon slowdown, startup, checkpointing, evaluation, contention, interruptions, scheduling and resource wait. A single 12h allocation is inadequate for 18 x 20M at either measured reference rate. Do not assume linear scaling to 20 trainers, nor reserve 64 CPUs/400G/one GH200 as a final profile just because that allocation was observed.

## 8. Implementation phases, ordered by dependencies

| Phase | Future deliverable | Acceptance gate before next phase |
| --- | --- | --- |
| **0 — Scientific specification** | Freeze 18 membership, single seed, explicit training budget, variants' intended direction/SL/TP/FSM/reward differences, evaluation split roles. | Research comparison matrix and budget approved; no ambiguous A0 vs A1. |
| **1 — Config and causal integrity** | Correct **proven** A0 inheritance defect if approved; effective-config snapshots; feature availability, train-only normalization, data splits, cache and entry-gate checks. | All 18 resolved with expected action/observation sizes; causal timestamp and split boundary tests pass. |
| **2 — Backend science** | Real-data frames/bundle differential traces, model-state/evaluation check; preserve established fixture and PPO reports. | No unexplained action/trajectory/reward/risk discrepancies; numeric tolerances and tested scope documented. |
| **3 — Recovery/provenance** | Harden requested-step completion checks, identity fingerprint, valid final checkpoint loading, collision-free outputs, retry logs and atomic state. | Partial job/corrupt zip/changed config/crash/restart cannot be called successful; all previous logs preserved. |
| **4 — CI and representative pilots** | Linux ruff format/check, mypy, targeted pytest, bash syntax; controlled one-per-strategy pilot, long enough to exercise eval/checkpointing. | No unresolved CI/scientific failures or violated risk-survival criteria; OOS remains locked. |
| **5 — Resource experiment** | Matched sustained 1x8, 1x16, 2x8 and 2x16 PPO profiles; higher concurrency only if evidence warrants. | Repeatable aggregate throughput and observed cgroup/process/GPU/disk peaks prove headroom for actual allocation. |
| **6 — Launch decision** | Immutable 18-key dry-run and manifests, multi-allocation restart schedule, requested Slurm resources, durable tmux/controller logs, final checklist. | Separate authorization to launch production and a valid available compute allocation. |
| **7 — Audit (after authorized training)** | All 18 full-budget runs verified, failed/partial attempts preserved, reproducible metrics and locked OOS reporting. | Precisely 18 complete runs or an explicit table of missing/failed runs; no unverified success. |

## 9. Tests and phase-specific acceptance criteria

**Config:** tests/test_train/test_variant_resolver.py and effective-config matrix checks: exactly 18 IDs and seed-42 keys, explicit budget, correct architectures, reward/risk/features, A0/A1 SL flags and constructed Box dimensions. Training logs now record all observation branch shapes to confirm A4 hidden state is absent. These checks still need execution.

**Causality/data:** focused MTF/HTF alignment, MTF variants, swings, sweeps, PO3 and IFVG tests passed **126/126** on 2026-10-14. A second suite passed **29/29** across causal-feature, split, frozen-normalization, cache/fit-scope, assembled-matrix truncation and leakage-detector tests (two expected Transformer warnings). A third suite passed **30/30** for the missing-entry-feature warning/fallback, strict eligibility wiring, rejection counters and golden environment traces. All used `.venv-x86`, Python 3.12.14, pytest 9.1.1; exact commands/results are in the implementation ledger. These are fixture/unit checks. The current-source real-market feature-prefix audit passed 9/9 cutoff comparisons. The current-source frames/bundle report passed all 18 variants over fixed full-calendar and active-action traces with exact state/observation/reward parity. This strengthens bounded evidence; the exposed 2026 interval remains unsuitable for pristine confirmation, and do not infer all-path causal certification from these checks.

**Environment science:** tests/test_envs/test_shared_bundle.py and fixed-action differential real-data tests for all 18. Assert exact discrete decisions and bounded floating-point differences in observations, rewards, trade/position state, direction, SL/TP modes and partial fractions, FSM transitions, PnL, equity, risk kills and done/truncated flags. On mismatch report first diverging timestamp/field/config. Include full policy state_dict keys, shape, dtypes, buffers and saved checkpoint integrity when comparing matched training.

**Artifacts:** tests/test_campaign_artifacts.py plus missing/corrupt zip, zero/partial timesteps, wrong data/config/git fingerprint, interrupted attempt, failure propagation, resume idempotency, file truncation, checkpoint loads, disk-full/preemption and stale .done scenarios. Require actual requested budget and correct status/exit code; no double schedule; attempt logs remain append-only.

**Execution:** on authorized compute node run ruff format --check, ruff check, mypy and targeted/integration pytest, bash -n for edited scripts. Capture exact commands and versions. Verify tmux detach/disconnect, persistent stdout+stderr logs, chronological attempt/Slurm records, sampled plus peak RAM/CPU/GPU metrics and original errors. **No tests were run while authoring this plan.** Historical 36-pass claim is not a new CI gate execution.

## 10. Estimates and unresolved measurements

| Quantity | Value | Evidence / limitation |
| --- | --- | --- |
| Core variants | **18** | Verified scripts/variant_list.txt; no knob/ablation extras |
| Campaign seed and budget | **one seed (42), 20,000,000 steps per variant** | Researcher-selected and frozen in config/campaigns/18_variant_seed42_20m.yaml; legacy registry defaults remain historical |
| Default steps conflict | **100,000** registry vs **20,000,000** default PPO/controller | Verified; resolved for this campaign by its explicit frozen spec |
| Incremental 16-env RAM | frames **6.16 GiB**, bundle **1.41 GiB** | Seven-day environment-only probe |
| 16-env process PSS | frames **10,010 MiB**, bundle **6,590 MiB** | Environment probe; not PPO peaks |
| Matched 32k-step PPO speed | frames **798.24**, bundle **1,029.87 steps/s** | Seven-day 8-env matched report |
| Concurrent 2x8 and 2x16 | **498.71** and **1,066.99 aggregate steps/s** | Short PPO reports; no scaling guarantee |
| 18-worker n_envs=64 probe CPU use | **0.35 average CPU cores per worker step**, about **5.8%** of each worker's 6-CPU assignment | Slurm `TotalCPU` for all 18 worker steps; 100k shared-bundle probe, not frozen frames/n_envs=4/20M shape |
| Four trainers | **No valid completed throughput** | Attempt terminated; root cause open |
| 18 x 20M at measured 2x16 rate | **~93.7 hours ideal** | Derived excluding overhead; not a walltime promise |
| Long-training process-tree PSS peak, cgroup RAM, CPU, GPU, scratch pressure | **Unknown** | Requires monitored sustained production-like PPO profile |
| 18-variant real-market frames/bundle equivalence | **Exact parity observed for tested traces** | Current-source report: 18 variants; 49,925-step test-calendar and 10,000-step active-action traces each, max scalar delta 0.0; source/raw-data hashes reverified. Not an exhaustive policy-state proof. |
| Pilot acceptance and fresh locked-OOS outcomes of new campaign | **Unknown** | Exposed seed-42 2026 outcomes are development evidence only; never infer final performance from them |

## 11. Exact files likely to change in a later authorized implementation

**Experiment and resolver:** config/experiments.yaml, quant_rl/config/default.yaml, quant_rl/train/variant_resolver.py; optionally a new frozen campaign spec. scripts/variant_list.txt remains the authoritative list unless the researcher changes scope.

**Scientific layer, only if a concrete fault is proven:** quant_rl/envs/trading_env.py, quant_rl/train/train_rl.py, relevant quant_rl/features/ modules and quant_rl/data/split.py. Verify source-specific logic before editing. Corresponding tests: tests/test_train/test_variant_resolver.py, tests/test_envs/test_shared_bundle.py, tests/test_causal_features.py, tests/test_data_split.py and focused action/risk/entry tests.

**Campaign/recovery:** scripts/matrix/campaign_controller.py, scripts/train/train_one_variant.sh, scripts/matrix/run_18variant_matrix.sh, scripts/matrix/run_18variant_matrix_gh200_tmux.sh and tests/test_campaign_artifacts.py; scripts/triton/activate_venv.sh only upon proven architecture issue. Consider updating docs/training/triton_gpu_training_agent_plan.md and docs/training/triton_ram_optimization_plan.md in a **separately authorized** documentation synchronization step. This plan does not modify any of them.

**Relevant references:** AGENTS.md; .agents/domain_maps/training.md; .agents/rules/triton-slurm.md; scripts/variant_list.txt; config/experiments.yaml; config/features/full_po3_mtf.yaml; quant_rl/config/default.yaml; quant_rl/train/variant_resolver.py; quant_rl/train/train_rl.py; quant_rl/envs/trading_env.py; scripts/matrix/campaign_controller.py; scripts/train/train_one_variant.sh; outputs/gpu_campaigns/ram-validation-20261010/reports/. All described current findings are based on this repository/reports, not a Windows clone.

## 12. Launch-readiness signoff checklist

- [ ] All 18 variants have an approved scientific hypothesis, unambiguous A0/A1 meaning and recorded intentional differences.
- [x] For the frozen GPU-balanced campaign, seed 42 and 20,000,000 steps/run are explicit in the campaign spec; the allocation launcher passes that spec without seed/budget overrides, the controller takes these values from the spec and rejects campaign-identity CLI overrides, and the dry-run confirms 18 seed-42 keys / 360,000,000 total steps. Legacy registry defaults remain historical and are not claimed to be globally reconciled. Evidence: `config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml`, `scripts/matrix/run_campaign_allocation.sh`, `scripts/matrix/campaign_controller.py`, and `outputs/gpu_campaigns/readiness_reaudit_20261014/production_shape_dry_run.json`.
- [ ] Git SHA, runtime/software versions, source/tick data hashes, time splits, training-fit masks, feature-cache hashes and final evaluation protocol frozen for the new campaign.
- [x] All 18 variants instantiated and actual action/observation dimensions checked; A0/A1 inherited flag issue resolved without changing risk criteria.
- [ ] Real-market feature causality, leakage tests, gold trace entry-gate warning and OOS isolation demonstrated; no look-ahead.
- [x] All 18 real-data frames-versus-bundle parity findings recorded for fixed data and action traces, including trade/risk/trajectory state and numerical deltas. See the scope limits above.
- [x] Historical full Linux suite passed 1,397 tests (2026-10-13); current worktree changes have the scoped regression results recorded in the ledger, and a full-tree rerun is not claimed.
- [x] Controller admission enforces the expected timestep budget, CPU-loadable checkpoint integrity/schema, campaign/run/attempt provenance, required metrics, zero worker exit status and the training-equity gate. Source review plus `tests/test_campaign_artifacts.py` (17/17 recorded) verify the contract. This closes the admission-control implementation check only: the 100k pilot's 18 failed equity gates mean it produced no successful campaign runs. Evidence: `scripts/matrix/campaign_controller.py`, `tests/test_campaign_artifacts.py`, and the 2026-10-14 campaign-artifact test result in `docs/thesis/implementation_ledger_2026-10-13.md`.
- [ ] Durable tmux, worker/controller/resource logs survive disconnect and preserve prior attempts.
- [ ] Representative sustained PPO and concurrent profiling supports actual n_envs, worker count, CPU/RAM/GPU limits, allocation duration and scratch quotas; measured peaks + headroom documented.
- [ ] Independent validation/selection and fresh locked final OOS roles respected; exposed 2026 development results remain non-confirmatory; risk-survival failures remain blockers. Legacy A2/A3 archives are excluded per the linked decision record.
- [x] Dry-run of the frozen GPU-balanced spec shows exactly 18 unique seed-42 keys, correct output root, frames backend, 20M steps each (360M total), 18 workers, `n_envs=4`, 3 CPUs/worker and two GH200s; no worker was started. Evidence: `outputs/gpu_campaigns/readiness_reaudit_20261014/production_shape_dry_run.json`.
- [ ] Production/confirmatory launch receives explicit authorization, an available allocation, and passing scientific/readiness gates. Those conditions are not met. A separate exploratory, selection-acknowledged 18 × 20M run is active under `config/campaigns/18_variant_seed42_20m_purelegacy_shared_bundle_n64_20261014.yaml`; its existence does not approve production or confirmatory use.

**Current action priority:** the current-source feature-prefix and frames/bundle parity audits are complete for their documented bounded scopes; preserve their hash-stamped reports. The frozen GPU-balanced 20M dry-run validated 18 seed-42 keys and 360M total steps without starting workers. The 100k 18-variant pilot failed all 18 training gates; preserve that result and do not rank variants. The frozen 20M campaign has not launched. Remaining work is sustained trainer resource sizing, independent research review, a scientifically justified resolution of the strategy acceptance failure, and preserving the exposed-data boundary. A pristine holdout is required only for confirmatory claims, not the acknowledged exploratory work. Do not launch the 20M campaign while its strategy acceptance gate has no passing evidence.

**Historical parity-job checkpoint — 2026-10-14 10:34 local:** the full-market CPU-only frames/shared-bundle audit was running as Slurm job `20951231` (4 CPUs, 128 GiB, 12 h) on `milan20` in `batch-milan`; the progress JSON showed `ladder_a0_flat` processing. The job subsequently completed successfully; the validated report is recorded below.

## Bounded exploratory run update — 2026-10-13

The explicitly requested A0/A1 development comparison has now run on one H100
worker for 100,000 requested PPO steps per variant, seed 42. The configured
2026 evaluation period is previously inspected and selection-acknowledged.
Both variants failed test survival and the in-sample training-equity gate;
their controller status is failed, and no candidate was selected. Complete
reports and checkpoints are preserved in
`outputs/gpu_campaigns/dev_a0_a1_seed42_100k_exportfix_exposed_2026/`, with a
hash-indexed summary at `reports/exploratory_comparison.json`. The prior
incomplete export attempts are retained separately. The comparison is
non-confirmatory and does not close the need for a new final holdout.

This bounded run adds no approval for the 18-variant production matrix. Its
readiness checklist remains open: obtain a fresh post-2026-07-03 holdout,
complete real-data causal/leakage audits, and establish sustained resource
sizing for the chosen concurrency. All-variant frames/bundle parity and
campaign failure/resume checks are now complete. The missing A2/A3 archives are
formally excluded by the linked decision record. Seed-only bitwise
repeatability was not demonstrated; A0 repeated under the same nominal seed
produced different metrics.


**Continuation audit update — 2026-10-13:** the full real-market prefix
invariance audit was rerun through five cutoffs for each of the baseline,
PO3/IFVG and distribution feature profiles, using three CPU-only workers on
Slurm job 20937646. All **15/15** comparisons passed with zero changed
features. The merged report and individual hash-indexed reports are in
`outputs/gpu_campaigns/real_market_feature_causality_20261013_resume/`. The
final full pytest rerun passes **1,397 tests, 38 warnings**; details and logs
are in `docs/thesis/implementation_ledger_2026-10-13.md` and
`outputs/gpu_campaigns/readiness_verification_20261013_resume/logs/full_pytest.log`.
This is feature-prefix evidence on the exposed data available through
2026-07-03. It does not certify all execution/reward/selection paths and does
not support confirmatory claims. Per the researcher’s explicit scope
clarification, current available data is accepted for exploratory continuation;
a new post-July holdout remains unavailable and is not required for this
exploratory work. Production readiness remains **NOT READY** because pilot
equity/survival gates fail, sustained resource-sizing and independent research
reviews remain open, and exposed outcomes cannot serve as confirmatory
validation.

**Researcher-requested output reset and exploratory 18-run continuation —
2026-10-14:** the researcher asked to clear the generated output area and
restart. All contents of `outputs/` were deleted, including prior checkpoints,
logs, audit reports and figures; no archive was retained. The previous
Slurm job `20938752` had already been confirmed cancelled after A0 reached
approximately 4.03M of 20M requested steps. That campaign was incomplete, and
its partial checkpoint is no longer available. Earlier output paths in this
plan now refer to deleted files; the implementation ledger records which
historical checks were reported before the purge.

A new exploratory campaign is defined by
`config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_20261014.yaml`:
18 variants, seed 42, 20M steps each, frames backend, selection acknowledged
on the previously exposed data. The intended allocation is two distinct
GH200 nodes with one GPU, 64 CPUs and 900G RAM per node; workers are pinned
as nine concurrent trainers per GPU. `scripts/matrix/run_campaign_allocation.sh`
and `scripts/matrix/campaign_controller.py` implement this placement. The
controller records worker node/PID provenance, saves controller output, and
starts one resource sampler per node for worker-tree memory/I/O and local GPU
telemetry. These additions pass the campaign artifact, sampler, year-survival,
golden-trace and deterministic-checkpoint suites (34 tests) with scoped Ruff
checks. The
18-run dry-run lists exactly 18 runs/360M total steps; shell syntax and Python
compilation pass. Slurm allocation request `20942600` was cancelled because it
exceeded the `aalto_users` GH200 group run-minute limit. A six-hour test-only
request for the same two-node shape is legal and forecasts start at
2026-10-16 13:32:18 on `gpuarm1` and `gpuarm2`; this is not a reservation. No
trainer has started.
The user's exploratory authorization allows running this study on available
data, but readiness remains **NOT READY**:
the A0 pilot failed its equity/survival gates, independent research reviews
and sustained resource sizing remain open, and exposed outcomes are not
confirmatory evidence.

**Training/evaluation breach penalty alignment — 2026-10-14:** review of the
reward path showed episodic training assigned zero terminal reward for a hard
risk breach while evaluation assigned `-10.0`. Training now uses the same
`-10.0` penalty; the independent train equity/survival gate is unchanged. The
diagnostic reports a train/evaluation penalty mismatch if these diverge.
Verification passes in the 14-test year-survival and golden-trace suites, with
scoped Ruff checks. This change has not yet been exercised by full training;
the current Slurm campaign request is still pending.

The one-rollout `ppo_best` diagnostic now uses deterministic actions. In this
campaign it runs once at the final training timestep, so it records the final
policy on a training-calendar slice rather than selecting among intermediate
checkpoints. `tests/test_train/test_checkpoint_eval_determinism.py` covers
the deterministic-action contract.

**Current continuation status — 2026-10-14:** the requested exploratory
campaign is authorized but has not started. The output directory is empty and
there are no active Slurm jobs or trainer processes. A second controller defect
was corrected: artifact hashing now returns the collected hashes, so failed
attempt evidence is retained. PPO continuation now preserves the checkpoint's
critic/value head instead of re-zeroing it after load. Resume tests confirm
completed failures remain terminal, and the checkpoint selector requires
matching attempt provenance and campaign fingerprint. Campaign identity hashes
runtime source/config inputs, not unrelated documentation, so ledger/status
updates do not invalidate a resumable campaign. The 18-way supervisor
(`scripts/matrix/run_campaign_18way_supervisor.sh`) requests sequential legal
six-hour allocations with a bounded retry for allocation failures; it retries
only pending/interrupted runs and exits on terminal failures. The planned nine
workers per GPU, each with eight environments, remain an unmeasured concurrency
profile; campaign telemetry must be reviewed before calling that sizing
validated. Focused campaign, resume, sampler and trainer tests pass (33 tests),
as do shell syntax, Python compilation, scoped Ruff and `git diff --check`. The
output purge removed the historical reports referenced earlier in this plan;
their values are documented historical notes, not currently re-openable
evidence. Overall thesis/production readiness remains **NOT READY**: the A0
pilot failed its equity/survival gates, independent research review remains
open, and exposed data cannot support confirmatory claims. The researcher has
waived a fresh holdout only for this acknowledged exploratory campaign.

**GPU placement correction — 2026-10-14:** the first actual 18-worker launch,
job `20943263`, landed on one GH200 despite requesting two. Inspection of
`CUDA_VISIBLE_DEVICES`, Slurm step accounting and live device UUIDs found every
worker pinned to physical GPU 0; GPU 1 was idle. The run was stopped before a
checkpoint or completed training result. Its 18 attempts are recorded as
interrupted and preserved with the pre-stop manifest and reconciliation report
under `outputs/gpu_campaigns/ladder18_seed42_20m_exploratory_selection_acknowledged_18way_20261014/reports/`.

The launcher now assigns workers evenly with explicit `map_gpu:0/1` binding;
the same-allocation probe resolved to distinct GH200 UUIDs. It requests both
allocated devices per overlapping worker step and binds that worker to one;
the sampler runs in a CPU-only step. A new campaign spec and output identity
preserve the frozen 18 variants, seed 42, 20M steps, frames backend and
selection-acknowledged exploratory status:
`config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml`.
The intended shape is 18 workers (9 per GPU), `n_envs=4`, 3 CPUs per worker,
and one `gpuarm1` allocation with 64 CPUs, 1000 GB RAM and two GH200s. The
controller command tests pass (17 tests), as do scoped Ruff, shell syntax,
Python compilation and an 18-run/360M-step dry-run. The corrected run is still
exploratory; the separate readiness gates and independent research reviews
remain open, and the system remains **NOT READY** until their evidence is
reviewed.

The corrected supervisor is active in tmux session
`thesis-18way-gpu-balanced-20261014`; job `20943576` has started on `gpuarm1`.
The campaign manifest reports 18 active workers, assigned 9 per GPU, at
`n_envs=4` each. At 49 seconds, startup was still in progress: no worker had
entered PPO, cgroup memory was 87,814,737,920 bytes, CPU usage was 142,299,279
usec, and both GH200s were at 0% with 3 MiB used each. This early snapshot
does not establish sustained resource sizing. The preceding test-only estimate
of 2026-10-16 13:52 EEST was not a reservation; the actual allocation started
on 2026-10-14.

**Latest campaign and reporting checkpoint — 2026-10-14:** the earlier
18-worker `n_envs=4` launch under job `20943576` is historical. A separate,
explicitly exploratory resource probe is now running as job `20944844` in tmux
`thesis-organizer`, window `probe18`: 18 variants, seed 42, 100,000 requested
steps each, `n_envs=64`, shared bundle, 6 worker CPUs, 2 GH200s, 128 allocated
CPUs and 900 GB RAM on `gpuarm2`. At the latest controller poll, 16 workers
were still evaluating/exporting and two had failed the in-sample training
equity gate (`ladder_a1_sl_po3`, `ladder_a4b_po3_ifvg`). Preserve both failures
as outcomes; do not resume them as new attempts or treat this probe as the
frozen 20M thesis campaign.

The old resource sampler captured 62 process-tree samples through
00:58:01 UTC: PSS mean 238.6 GiB, peak 251.2 GiB; reconstructed per-GPU sampled
utilization averaged 65.8% and 66.5%, peaking at 92% on both. The sampler's
cgroup counters refer to its own step and are not allocation memory, and this
sampler version did not capture process-tree CPU time. These are partial
resource observations while final replays were active; resource sizing remains
open until terminal job/accounting evidence is reconciled. The live allocation
is to remain untouched.

Dashboard/report changes now account for unlisted close reasons, label the
reward PnL term as a reward contribution, distinguish the periodic
training-prefix monitor from the two final split replays, separate plots by
metric units, and emit split-specific evaluation JSON plus a readable summary
with PnL and account currency. Ruff format/check passed, and the focused
dashboard, evaluation artifact and sampler suites passed 35/35. The current
workers started before these source edits and keep the earlier output format.
These implementation and probe results do not change the decision: production
readiness is still **NOT READY**. The old A0/A1 pilots fail acceptance; the
18-worker run is a 100k resource probe rather than the frozen 20M campaign;
independent research sign-off, other research acceptance reviews, and
confirmatory holdout evidence remain open. The researcher's no-post-July-data
waiver continues to apply only to exploratory work.

**Standard MTF configuration checkpoint — 2026-10-14:** extended Idea 1/2's
confirmed PO3 sweep-reference config from hard-coded M5/M15/H1 to the standard
resampled HTFs M5/M15/M30/H1/H4/D1, preserving M5/M15/H1 as both Idea defaults.
Structure MTF features were verified with all six timeframes. PO3 pivot
confirmation now uses the next resampled timestamp and a local calendar-day
boundary for D1 across DST. The focused feature tests pass 44/44 and scoped
Ruff format/lint pass. This closes the standard-timeframe configuration gap;
it does not certify full-history real-data causality for every feature profile
or show that adding timeframes improves returns. Any changed feature config
gets a different content-hashed cache. Production remains **NOT READY** under
the open gates above.

**Failed-attempt resume contract re-audit — 2026-10-14:** current controller
source preserves `failed` jobs as terminal; only `interrupted` attempts return
to the pending queue. `tests/test_campaign_artifacts.py` verifies failed exit
codes, attempt numbers and artifact hashes remain unchanged across resume
(17/17 passed). This supersedes the older resume-risk note for the current
controller source; allocation-boundary and multi-allocation resource evidence
remain separate open gates. Do not retry the two scientific gate failures in
the live 100k probe.

### Live probe update — 2026-10-14 01:09 UTC

Slurm job `20944844` remains RUNNING on `gpuarm2` (2 GH200s, 128 CPUs, 900 GB
requested RAM). The controller manifest now reports **3 failed, 15 running,
0 succeeded**. The three seed-42 workers failed the train-equity gate; retain
these outcomes and do not retry them under this campaign identity:

| Variant | Train return / net PnL | Max drawdown | Sharpe | PF | Win rate | Trades | Gate finding |
|---|---:|---:|---:|---:|---:|---:|---|
| `ladder_a0_flat` | -2.67% / -$2,670.60 | 6.72% | -0.364 | 0.875 | 39.3% | 89 | End below start, negative slope, max-loss stop |
| `ladder_a1_sl_po3` | -4.96% / -$4,956.43 | 7.13% | -0.642 | 0.802 | 34.5% | 58 | End below start, negative slope, max-loss stop |
| `ladder_a4b_po3_ifvg` | -1.84% / -$1,841.15 | 1.84% | -1.249 | 0.000 | 0.0% | 8 | End below start and negative slope |

These are in-sample training replay metrics for only three variants. The other
15 workers have no terminal outcome yet. Evidence is in the worker logs and
each attempt's `training/metrics.json` and `train_equity_gate.json` under
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/runs/attempts/`.

The resource CSV contains 78 samples from 00:14:46 to 01:09:13 UTC. Aggregate
process-tree PSS averaged 235.9 GiB and peaked at 251.2 GiB. The file has a
16-column header without process-tree CPU time, so CPU use cannot be recovered
from this live sampler. The checked-in sampler source has a CPU-time counter,
but the running sampler loaded the earlier version. Its cgroup fields describe
the sampler step, not allocation memory. This does not validate 900 GB as a
minimum or establish the CPU requirement. Leave the live job untouched and
reconcile all results and terminal Slurm accounting after exit.

Slurm `sstat` supplies a separate cumulative CPU-time estimate for the 15
currently running worker steps. Dividing each step's `AveCPU` by its elapsed
time gives **0.27–0.34 CPU cores per worker on average** over roughly 56
minutes, or **4.4–5.7% of the six CPUs assigned per worker** (about 4.8 cores
in total across 15 workers). This is an interval average, not an instantaneous
sample; it points to CPU underuse during this probe window, not CPU saturation.
Retain the caveat that worker phases and runtime conditions can change, and
measure another sustained window before changing production allocation shape.

### Final-evaluation timing and deterministic gate — 2026-10-14 01:15 UTC

The 18 current worker logs report PPO-learning times for all 18 workers and
test-replay times for all 18. PPO learning had a 9.4-minute median (5.9–13.7
minutes); test replay had a 16.1-minute median (10.4–24.6 minutes). Seven
training-split replays had completed at this snapshot, with a 27.6-minute
median (18.6–31.8 minutes). These phase timings show final evaluation currently
costs more elapsed time than PPO learning in this 100k probe. They do not
establish a universal production forecast. Workers are still saving/exporting
reports, so terminal controller state remains authoritative.

The current checkout before this correction used deterministic actions for
the final test replay but stochastic actions (`deterministic=False`) for the
full training replay that feeds `train_equity_gate.json`. The live attempts
record a different `train_rl.py` source hash from the current checkout, and
their source snapshot is not retained, so their exact action flags cannot be
verified from the current file alone. Changed both final split replays to use
one deterministic helper; the periodic dashboard progress monitor remains
stochastic. Split JSON now records `policy_action_mode: deterministic`, and
the Markdown summary states both final replays use deterministic policy
actions. The focused checkpoint-determinism and report-artifact suites pass
**15/15** combined; scoped Ruff format/lint passes. This does not retroactively
change or recertify the live workers. Any future run must use a new
fingerprinted campaign identity and the corrected deterministic gate.

### Interim outcomes update — 2026-10-14 01:16 UTC

The latest controller manifest reports **12 failed, 6 running, 0 succeeded**
for job `20944844`; no workers are pending. Fourteen worker logs already
contain a failed `TRAIN_EQUITY_FAILED` record. Twelve of those workers are
terminal in the manifest; `ladder_a4a_entry_fsm_hidden_dist` and
`ladder_a4a_entry_fsm_hidden_po3` have written failed gate records but remain
running while artifacts are saved. Do not retry them.

All 18 worker logs now contain a final test replay line. Twelve variants did
not survive the 130-day test period:
`ladder_a0_flat`, `ladder_a0_flat_dist`, `ladder_a0_flat_po3`, `ladder_a1_sl`,
`ladder_a1_sl_dist`, `ladder_a1_sl_po3`, `ladder_a2_tp`, `ladder_a2_tp_dist`,
`ladder_a2_tp_po3`, `ladder_a3_multi_tp`, `ladder_a3_multi_tp_dist`, and
`ladder_a3_multi_tp_po3`. The other six survived but placed at most two trades:
two had zero trades, four had one or two trades with returns between -0.10%
and -0.51%, and none provides an active positive full-period result. These
are report-only outcomes on the previously inspected split; they are not
confirmatory evidence. Source evidence is the per-variant `[test]` line in
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/logs/workers/`.

The six workers still running are `ladder_a3_multi_tp`,
`ladder_a4a_entry_fsm_hidden`, `ladder_a2_tp_po3`,
`ladder_a4a_entry_fsm_hidden_po3`, `ladder_a4b_distribution`, and
`ladder_a4a_entry_fsm_hidden_dist`. Several are completing train-split report
export. Wait for Slurm and controller terminal state before treating this
snapshot as final. The probe has produced no basis to select a strategy; the
production campaign remains **NOT READY**.

**Near-terminal update — 2026-10-14 01:17 UTC:** the controller manifest now
reports **17 failed, 1 running, 0 succeeded**. The only active worker is
`ladder_a3_multi_tp`, which has completed its test replay and is in its
full training replay. The 17 terminal workers have train-equity gate failure
records. The Slurm allocation is still RUNNING; do not treat the 18-run probe
as complete until that worker exits and the controller records its terminal
status.

### Final resource-probe reconciliation — 2026-10-14 01:25 UTC

The campaign controller is terminal: **18 failed, 0 interrupted, 0 pending,
0 running, 0 succeeded**. All 18 workers have a saved train-equity gate and
training/testing metrics; all 18 failed the train-equity gate, including all
three A0 variants. The report-only test replays did not establish a viable
variant: 12/18 failed the 130-day survival period; the other six survived with
zero to two trades. The complete per-variant results and SHA-256 evidence are
in `outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md`
and `.json`. The report was regenerated from the campaign manifest, controller
status, metrics, worker logs, provenance and resource samples; its focused
report-builder suite passes (**2/2**) and Ruff check passes. The report now
reconciles each saved split's reported `total_pnl` against final equity minus
initial balance: **36/36 split records match within $0.02**, with zero missing
or mismatched values.

Slurm job `20944844` remains **RUNNING** on `gpuarm2` with 128 CPUs and 900 GB
requested memory, but the terminal sampler row at 01:19:14 UTC records no
campaign worker processes (`process_tree_count=0`, `campaign_active=0`). Per
worker Slurm accounting shows 0.35 CPU cores average over each worker's full
step (range 0.31–0.47 cores, about 5.9% of the six CPUs assigned). Across 95
resource samples, process-tree PSS averaged 218.9 GiB and peaked at 251.2 GiB;
each GPU averaged 66.4% utilization, with peaks of 97% and 96%. This probe
does not measure allocation-wide cgroup memory and its sampler did not record
process-tree CPU time. The CPU estimate is an elapsed-time average, not an
instantaneous demand profile. Leave the existing Slurm allocation in place as
requested; do not cancel it. No trainer is active according to the final
sampler row, and all 18 worker steps are terminal in `sacct`.

This was an exploratory **100k-step resource probe**, not the frozen
20,000,000-step thesis campaign; the latter has not launched. The report-only
test interval was previously inspected and remains exposed development data.
The researcher authorized use of data through 2026-07-03 for exploratory work,
so acquisition after that date is not a prerequisite for this probe; an
untouched final holdout is still required before confirmatory claims. Independent
research sign-off, complete execution/reward/termination and model-selection
acceptance, a passing strategy gate, and production resource sizing remain
open. Readiness remains **NOT READY**. Do not rank candidates or weaken gates
based on this one-seed result.

### 3-strategy run interruption reconciliation — 2026-10-14 01:30 UTC

The separate three-worker 20M exploratory run
`ladder3_strategy_seed42_20m_exploratory_selection_acknowledged_20261014`
did **not** finish. Its raw controller snapshot still says 3 running, but
Slurm job `20944286` is terminal FAILED after 35 minutes; the allocation
supervisor log records the `srun` step aborted by SIGKILL, and all three
worker steps are CANCELLED in `sacct`. The controller snapshot is preserved
unchanged. Do not call `--resume` on this campaign as a status check: the
controller turns interrupted jobs back into pending work and would launch
them again.

The last dashboard rows reached 704,512 steps for baseline and PO3 and
606,208 for distribution, far short of 20M. Those rows contain only progress
replay return/trade counts; no account-currency PnL column was saved. None of
the three runs has `training/metrics.json`, `testing/metrics.json`, or a final
`evaluations/summary.md`. Thus there is no final train/test PnL for this run.
The evidence-backed reconciliation is
`outputs/gpu_campaigns/ladder3_strategy_seed42_20m_exploratory_selection_acknowledged_20261014/reports/campaign_reconciliation.md`
and `.json`, with saved Slurm accounting in the same `reports/` directory.
Keep this interrupted 20M run distinct from the completed 100k 18-worker
resource probe. Neither provides a successful candidate result; readiness
remains **NOT READY**.

### Retained allocation state recheck — 2026-10-14 01:37 UTC

Read-only `squeue`/`sacct` recheck confirms job `20944844` is still RUNNING on
`gpuarm2` with 128 CPUs and 900G requested memory. The allocation wrapper step
`.0` remains RUNNING, while the 18 worker steps `.7`–`.24` are terminal
FAILED (exit 1); the earlier three worker steps `.2`–`.4` are CANCELLED. The
latest resource sample remains `2026-10-14T01:19:14.975605Z` and records zero
campaign processes. Thus this is a retained allocation, not active training.
Leave it in place per the researcher's instruction; do not cancel or start a
replacement step.

The checksummed 18-row report was regenerated after adding direct worker-log
and run-artifact links to each row:
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md`
and `.json`. Split PnL still reconciles against saved final equity for all
36/36 records. This is reporting and state reconciliation only; it does not
change the failed strategy gates, exposed-data status, or **NOT READY**
decision.

### 100k probe gate-reason breakdown — 2026-10-14 01:40 UTC

The regenerated report now counts saved train-equity gate reasons (counts
overlap): `end_not_above_start` **17/18**,
`slope_not_positive` **15/18**, and `max_loss_stopped_trading` **12/18**.
This is a diagnosis of the existing 100k exploratory probe, not evidence of a
controller defect or a reason to relax the gate. PnL/equity still reconcile
for 36/36 splits; the frozen 20M campaign remains unlaunched and readiness
remains **NOT READY**.

### Probe source-fidelity audit — 2026-10-14 01:43 UTC

Compared all 18 attempts' recorded source inventories with the current
checkout. Each carries the same 194-file inventory; 186 current files match,
and eight have changed hashes (`config/2_po3_ifvg.yaml`,
`config/3_distribution.yaml`, `quant_rl/config/default.yaml`,
`quant_rl/eval/training_plots.py`, `quant_rl/features/build.py`,
`quant_rl/features/htf_sweeps.py`, `quant_rl/train/dashboard.py`, and
`quant_rl/train/train_rl.py`). No files are missing. The artifacts preserve
launch-time hashes and dirty-file hashes but not the prior source contents, so
exact retraining from this checkout is not established. Per-split metrics and
equity reconciliation remain inspectable. Repeatable audit outputs are
`outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/source_fidelity_audit.md`
and `.json`; the campaign summary links to them. This narrows the probe's
reproducibility claim and does not change **NOT READY**.

### Report clarity and live feature audit checkpoint — 2026-10-14 02:44 UTC

The A0 PO3 100k run now has a per-run `evaluations/summary.md` linked directly
from the campaign table. It distinguishes the single short train-prefix
monitor from the final train and report-only test replays and shows account
currency PnL. The early `115 closes` dashboard snapshot records 95.7% (about
110) structural stops; its old schema omitted labels for the other five
(4.3%), so their exact reasons cannot be recovered. This is not a zero-lot
finding: all 81 saved in-sample replay entries have nonzero lots. The final
replay remains train +$2,141.26 with a failed equity gate and report-only test
-$6,213.34; the 100k run predates the current clearer dashboard output.

A real-market prefix-causality re-audit is currently active on login CPU
(PID 877558; command and checkpoint are in the implementation ledger). It
completed the first baseline-profile cutoff comparison and is rebuilding the
March 2026 prefix. This is not a final result yet. Do not start the full
18-variant frames/shared-bundle parity audit until it finishes. The 18-run
probe still failed all 18 train-equity gates, the 20M campaign remains
unlaunched, independent research review and final-holdout evidence remain
open, and overall readiness remains **NOT READY**.

**Live prefix-audit recheck — 2026-10-14 03:52 UTC:** the same CPU audit
process/session (PID 877558 / session 91767) remains active after about 2h06m.
The baseline profile's three cutoff checks completed; the PO3 full-history
feature build is the last emitted stage. No final report exists yet. The
retained Slurm allocation remains RUNNING with no campaign worker active.
Continue the current process and run real-data frames/shared-bundle parity
only after it completes.

### Walk-forward model-selection action-mode correction — 2026-10-14

The canonical purged walk-forward path is used for architecture/hyperparameter
selection under `docs/thesis/validation_protocol.md`. Its fold evaluation now
uses the deterministic helper shared by final train/test reports; the prior
sampled-action path could make fold metrics depend on RNG draws. The existing
determinism/checkpoint callback test passes 2/2, while the walk-forward call
site itself was verified by source inspection. No fold training or new
selection evidence was produced. This implementation correction does not
change prior campaign outcomes, complete independent research sign-off, or
make any strategy pass its equity gate. The live feature-prefix causality
audit and subsequent frames/shared-bundle parity audit remain open. Readiness
remains **NOT READY**.

**Live audit/allocation checkpoint — 2026-10-14 05:35 UTC:** prefix-audit PID
`877558` / session `91767` is still CPU-active after 3h49m, building/comparing
the distribution profile. Baseline and PO3 profile comparisons have completed;
the final JSON is still absent. Slurm allocation `20944844` remains RUNNING
with only supervisor step `.0`; the three earlier trainer steps were
cancelled, the 18-worker probe failed, and no trainer step is active. Guard
session `62783` is waiting for natural allocation return before stopping the
login supervisor. Do not restart the audit or cancel the allocation.

**Checkpoint reanalysis mode and live-handle recheck — 2026-10-14 05:39 UTC:**
fixed the separate `eval_run` CLI mismatch: its help described stochastic
sampling although the evaluator default was deterministic. The CLI now makes
deterministic policy actions explicit by default, supports sampled actions only
with `--stochastic`, and records the selected action mode and seed in the output
manifest and training log. Focused provenance tests pass 15/15; scoped Ruff
format/lint and whitespace checks pass. No checkpoint was replayed. The live
feature-prefix audit remains active as PID `877558` / session `91767` with no
final JSON; allocation `20944844` remains RUNNING with no trainer steps, and
guard `62783` remains live. Real-data feature certification, independent
research review, a fresh untouched holdout and all strategy gates remain open;
the 18-variant 20M campaign is still unlaunched. Readiness remains **NOT READY**.

**Natural allocation timeout — 2026-10-14 05:43 UTC:** retained allocation
`20944844` reached its six-hour walltime and exited `124`; it was not
cancelled. Watchdog evidence confirms the login supervisor was stopped after
the `srun` returned, so it cannot silently allocate again. `squeue` is clear
for this job and there are no active trainer steps. Feature-prefix audit PID
`877558` / session `91767` remains CPU-active; it has not written its final
report. Its RSS is about 10 GiB in the latest check, with about 72 GiB
available on the shared login node. Preserve and monitor the existing audit;
do not restart it. Readiness remains **NOT READY** pending the audit, subsequent
frames/shared-bundle parity, independent review and unresolved scientific gates.

**Live risk source-review correction — 2026-10-14 05:52 UTC:** found and
corrected two concrete live-order risk defects. The RL and multi-symbol robots
previously sized orders against a stop distance but sent `sl=None`; they now
attach the broker stop computed from the sizing distance. `RiskManager` now
measures aggregate exposure from open-position entry-to-stop risk instead of
floating PnL, rounds lot size down, and blocks new orders if an existing stop
or account position query is unknown. Class-level default risk ratios match
the configured 1% per-position / 2% aggregate limits, and `RLRobot` itself now
defaults to paper mode. The complete `tests/test_live` package passes 47 tests;
scoped Ruff and whitespace checks pass.

This closes only the live-stop/sizing inconsistency. Live policy parity remains
open: the bridge emits direction-only signals and uses a fixed 50-pip stop,
not the training policy's structural stop and optional target/SL modes; live
does not implement training/evaluation cumulative daily and max-loss
guardrails. No MT5 broker integration was available, so keep `PAPER_TRADING`
true and do not treat the live path as production-ready. The 18-run training
campaign's **NOT READY** decision is unchanged; real-data causality audit and
the independent research/holdout gates remain open.

**Readiness recheck — 2026-10-14 05:59 UTC:** the saved shared-bundle resource
probe manifest and checksummed report confirm all 18/18 variants are terminal
failures; no variant is pending or running. The frozen 18-way 20M controller
dry-run still resolves exactly 18 unique seed-42 keys and 360M total steps.
Its configured shape is one `gpuarm1` node with two GH200s, 18 workers, four
environments and three CPUs per worker. This is not a validated sustained
20M profile, and the dry-run starts no job. The campaign remains unlaunched.

The separate three-variant shared-bundle 20M manifest has three interrupted
attempts; allocation `20944844` ended naturally with exit 124. Its tmux window
is an idle bash shell, and `squeue` is empty. No attempt was resumed or job
cancelled in this recheck.

Current scoped verification also passes the live suite (47 tests, two known
fixture warnings), the higher-timeframe/PO3 feature suite (117 tests), and the
campaign admission/result/source-fidelity/resolver/manifest suite (38 tests),
the execution/reward/survival suite (107 tests), and dashboard/final-report
contracts (45 tests), and data-split/normalization/walk-forward checks (28
tests), with scoped Ruff and whitespace checks. The real-market feature-prefix
audit remains live as PID `877558` / session `91767`; it is CPU-active and has
not written its final JSON. Keep full-market parity queued behind that audit. The
18/18 failed train-equity gates, exposed-data-only status, independent research
review and exploratory-versus-confirmatory evidence boundary remain unchanged.
Readiness remains **NOT READY**.

### Campaign source-snapshot reproducibility — 2026-10-14 06:26 UTC

New campaigns now archive the exact source and config files hashed into their
identity in `source_snapshot.zip` before any worker starts. The initial
`campaign.json` records that archive's SHA-256 and file count; resume verifies
the archive. Focused campaign artifact/source-fidelity checks pass **20/20**,
scoped Ruff passes, and `git diff --check` passes. A test-worker controller
fixture exercised snapshot creation and resume verification; no Triton
campaign was created or launched. This improves reproducibility for future
campaigns; it does not recover source contents missing from the historical
100k pilot.

A direct helper-level check with the frozen 20M spec archived and verified all
196 source/config inputs (688,685 bytes); the temporary archive was discarded
and no campaign artifacts were created.

The real-market prefix audit remains active as PID `877558` / session `91767`
in its final distribution-feature prefix rebuild; its JSON is not yet present.
Keep full-market frames/shared-bundle parity queued until its result is
validated. All 18 pilot training gates failed, so the frozen 20M campaign
remains unlaunched and readiness remains **NOT READY**.

### Current-source real-market feature-prefix audit complete — 2026-10-14

The audit in `outputs/gpu_campaigns/readiness_reaudit_20261014/feature_prefix_causality.json`
completed for all 18 registered variants, grouped into 3 feature profiles, and
passed all 9 profile/cutoff comparisons. The tested cutoffs are 2025-12-15,
2026-03-09 and 2026-06-15; each comparison has zero changed features and
maximum absolute delta 0.0. The 32 recorded feature/data/config source hashes,
the audit-script hash, and the US100.cash/US500.cash raw M1 hashes were
recomputed and match the current files. The report covers 528,760 rows through
2026-07-03. This is evidence of prefix invariance only at those cutoffs; it
does not prove every execution, label, split or model-selection path is
leakage-free. Full current-source frames/shared-bundle parity was subsequently
completed; see the completion record below.
The 18-run 100k pilot still failed all training-equity gates, the frozen 20M
campaign remains unlaunched, and readiness remains **NOT READY**.

The parity audit was then hardened for reproducibility: it now hashes all
Python sources under `quant_rl/`, all YAML configs under `config/` and
`quant_rl/`, the project lock/config files and variant list, and records the
runtime package versions plus Git branch/HEAD/dirty paths. Its focused audit
contract tests pass 3/3 and scoped Ruff format/lint passes. The subsequent
CPU-only Slurm audit and its verified result are recorded below.

The current resource sampler also has an end-to-end CSV contract for
`process_tree_cpu_time_usec` and `cgroup_memory_scope`; the focused sampler
suite passes 8 tests. This validates what a future sample will record, not a
new allocation's actual CPU or job-wide memory use. The historical 100k probe
remains unable to provide those missing measurements.

### Current-source real-market frames/shared-bundle parity complete — 2026-10-14

CPU-only Slurm job `20951231` completed successfully (exit 0; 4 CPUs, 128 GiB
requested, 41:40 elapsed). The hash-stamped report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.json`
(SHA-256 `372e565c913aacd0c564e19f03503d749ee4d2b1a3988f8a9aac289ab3481044`).
The human-readable companion is
`outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.md`;
raw Slurm accounting is retained in
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/slurm_accounting_20951231.psv`.
All 18 registered variants passed. For each variant, the frames and shared-
bundle environments matched exactly over the 49,925-step exposed test-calendar
trace and the 10,000-step active-action trace: 36 traces total, zero failed,
and maximum scalar delta 0.0. The report covers 177,210 test-calendar M1 rows
from 2026-01-02 01:05 through 2026-07-03 19:59 (Europe/Helsinki). Its 166
recorded project-source/config hashes and four raw market-data hashes were
recomputed against the current worktree and all matched; the recorded dirty-
path set also matches. The allocation reported MaxRSS 26,734,688 KiB (about
25.5 GiB) and TotalCPU 40:25.5. These are parity-audit resource measurements,
not PPO trainer sizing evidence.

This closes current-source frames-versus-shared-bundle parity for the specified
test calendar and fixed traces only. It does not prove exhaustive policy-state
equivalence, full real-market execution/label causality, independent research
acceptance, strategy viability, or pristine OOS performance. The exposed 2026
interval remains report-only. All 18 runs in the 100k pilot failed the train-
equity gate; the frozen 20M campaign remains unlaunched and readiness remains
**NOT READY**.

### Pure legacy baseline specification — 2026-10-14

The researcher confirmed that `ladder_a0_flat` must be the pure legacy baseline.
Its registry now sets `strategy_actions=false` and
`agent_direction_control=false`, selecting the legacy `Discrete(20)` path and
the default feature profile. The other 17 core variants retain their existing
strategy-action settings. No prior output was changed or deleted.

The 100k pilot's saved A0 config had `strategy.name=baseline` but
`strategy_actions=true`; its training trade log contains 55,900
`manipulation_unconfirmed` rejections. This is a measured property of that old
configuration, not a diagnosis of the new pure baseline and not evidence of a
zero-lot sizing bug. Its 89 opened positions had lot sizes from 0.4622 to
84.4835 (median 12.9653); those values do not establish that sizing was
otherwise appropriate. Keep the pilot failure and PnL immutable, but do not
present it as the pure legacy baseline result.

The corrected A0 changes action schema and feature profile relative to A1-A4.
Therefore A0 is a reference point, not a one-factor control for the SL/TP/FSM
ladder. The existing all-18 backend parity and action-space audits used the
previous A0 semantics. Re-run the current-source A0 config/action audit and
frames/shared-bundle parity before relying on that evidence for the corrected
registry. Do not launch the frozen 20M campaign: the pilot's other 17 gates
still fail, and the pure baseline has not yet been trained. Readiness remains
**NOT READY**.

### Live campaign and resource update — 2026-10-14 10:30 UTC

Slurm job `20954895` remains RUNNING on two GH200 nodes (48 CPUs and 512 GiB
per node, six-hour limit). The main manifest has 16 running jobs and two
failed first attempts. `ladder_a0_flat_po3` and `ladder_a0_flat_dist` failed
before PPO because their Idea 1/Idea 2 prebuilt cache hashes did not match the
effective data/config hashes. The failed outputs are preserved; those variants
are running in the separate
`ladder2_cache_recovery_seed42_20m_shared_bundle_n64_20261014` campaign. Thus
18 variant workers are currently active across two manifests, while the main
manifest correctly retains its 16-running/2-failed history. Sixteen workers
had emitted PPO progress from 491,520 to 884,736 steps at the latest log
sample; the two failed attempts logged no training steps. No final evaluation
or completed result exists yet.

The resource samplers recorded 70–71 samples over approximately 39–40 minutes
after startup. On gpuarm1/gpuarm2, process-tree PSS averaged 91.9/118.7 GiB
and peaked at 98.8/128.7 GiB. Slurm-job cgroup memory averaged 135.8/126.6
GiB; sampled current-memory maxima were 151.2/134.9 GiB and recorded cgroup
high-water marks were 152.9/136.5 GiB, against 512 GiB per node. The cgroup
scope is `slurm_job`; all OOM event counters remain zero. Process-tree CPU
averaged 3.91/4.87 cores against 48 assigned per node. Sampled GPU utilization
averaged 56.2%/52.1%, peaked at 96%/92%, and GPU memory averaged 38.0/40.2
GiB with peaks of 43.3/45.7 GiB. This is early-phase sustained evidence for
the actual shared-bundle `n_envs=64`, 18-worker allocation. It does not
establish final training, checkpoint, evaluation, scratch-I/O or maximum
resource peaks. Low sampled CPU use does not establish that allocating more
CPUs would improve throughput. Evidence is in the two campaign
`logs/resources_gpuarm*.csv` files and worker logs.

The current repository HEAD is `6d13698`, while the main campaign identity is
pinned to `937bae5` and fingerprint
`b19cf865ae01a900c425b9e7ab45fd922b5ac14b54364171cca34574ea270556`. Do not
rewrite either campaign's fingerprint to resume. Before another allocation,
verify exact-source resume behavior and preserve checkpoints and both campaign
manifests. Readiness remains **NOT READY**: no candidate has completed
training and both final split evaluations; the exposed 2026 interval is not
confirmatory; independent research review and a pristine holdout remain open.

**Acceptance-policy provenance caveat:** the active workers use source commit
`937bae5`, which predates the researcher-approved policy in `16b13ae`. The
source diff confirms that `937bae5` requires positive end equity and equity
slope for the train gate and defaults to stopping after the configured
no-trade threshold; the approved policy treats PnL/slope/activity as metrics
and disables the no-trade early abort. Therefore the active run is an
exploratory run under the earlier acceptance semantics, even though its
campaign name and 20M budget remain as configured. Preserve its outputs, but
do not represent it as having trained under the updated acceptance policy.
Any continuation from its checkpoints must use its exact pinned source and
retain this caveat; evaluating under the newer policy requires a separately
identified campaign with the newer source and fingerprint.

To protect source fidelity for any later retry, a detached worktree was
prepared at `/scratch/work/nguyenl37/Aalto_MS_Thesis_pinned_937bae`. All 201
files in the campaign's `source_manifest.json` match commit `937bae5`; its
data, cache, output and virtual-environment paths point to the existing shared
artifacts. The allocation launcher now routes a future resume through that
worktree when the active checkout SHA differs from the campaign manifest.
This path has passed shell syntax and worktree/hash checks but has not yet
been exercised by a later allocation. It does not alter the currently running
job.

The separate cache-recovery campaign is not covered by that fallback. A
pinned-source dry-run of its spec fails the controller's in-repository spec
path check because the spec currently sits under a symlinked output directory.
Its partial runs must remain separate until a provenance-preserving recovery
or migration is prepared; do not claim they will resume automatically.

### Live campaign follow-up — 2026-10-14 10:46 UTC

Slurm job `20954895` remains RUNNING (about 1 hour 20 minutes elapsed). The
main manifest records 16 running jobs and the two preserved cache-mismatch
failures; the separate recovery manifest records two running jobs. All 18
current worker logs now show PPO progress from 786,432 to 1,179,648 steps
(median 983,040). Ten 1,000,000-step checkpoints exist across the manifests.
No final evaluation has completed.

The three named prebuilt cache sidecars now match the hashes expected by the
failed workers: baseline `656b50e064710408`, Idea 1
`75feaab2164f0ee9`, Idea 2 `c5c8b3d9e890a760`; the two strategy caches also
have their required `.htf_sweeps.parquet` files. Their parquet matrices remain
local, untracked data. This proves the current cache artifacts are usable by
the recovery workers; it does not retroactively make the two failed attempts
successful.

The resource CSVs cover 55.5/55.8 minutes. On gpuarm1/gpuarm2, process-tree
PSS averaged 93.9/121.1 GiB and peaked at 102.4/131.4 GiB. Slurm-job cgroup
memory averaged 134.4/129.2 GiB; sampled current maxima were 151.2/138.8 GiB,
and cgroup high-water marks were 152.9/140.7 GiB against 512 GiB per node.
All `max`, `oom`, `oom_kill`, and `oom_group_kill` event counters were zero.
Process-tree CPU averaged 3.88/4.88 cores. GPU utilization averaged
56.3%/54.6% (peaks 98%/96%); VRAM averaged 39.6/41.9 GiB (peaks 44.0/46.8
GiB). These remain early-run measurements, not final-phase resource bounds.

A primary-campaign dry-run under the pinned source resolved all 18 keys and
20M steps/variant. The separate recovery spec does not pass the pinned
controller's repository-boundary check through the shared-output symlink.
The launcher therefore resumes only the primary campaign; recovery-run
migration remains open. The active run still uses pre-`16b13ae` acceptance
semantics and must not be presented as following the updated policy.

### Live campaign follow-up — 2026-10-14 10:49 UTC

Slurm allocation `20954895` is still RUNNING on gpuarm1/gpuarm2, with 4 h 37 m
remaining at the latest query. The primary manifest has 16 running workers
and two preserved cache failures; the separate recovery manifest has two
running workers. The 16 current primary logs report 786,432–1,277,952 steps
(median 1,081,344); both recovery logs report 983,040 steps. These are partial
training records only. No full-budget checkpoint, final in-sample evaluation,
or test evaluation exists yet.

The final evaluation implementation does run deterministic replays on both
the complete one-year training split and following test split and writes
separate `evaluations/train_in_sample.json` and
`evaluations/test_report_only.json` reports. The configured split is training
through 2025-12-31 and test through 2026-07-03. This satisfies the requested
evaluation coverage once training finishes; no current run has reached that
stage.

The allocation manager has a recovery continuity gap: the two recovery runs
are in their own manifest and were started in a separate Slurm step. The
durable two-node supervisor currently reads only the primary manifest, and its
allocation launcher resumes only that campaign. If this allocation ends
before recovery finishes, those two workers will need an explicit
provenance-preserving resume path; the main supervisor alone does not resume
them. Do not stop or rewrite either live run. Also retain the acceptance
policy caveat: these workers use pinned source `937bae5`, before the approved
policy in `16b13ae`.

### Checkpoint-preserving policy migration prepared — 2026-10-14

Added `scripts/matrix/build_campaign_checkpoint_map.py` and explicit initial
checkpoint handling to `campaign_controller.py`. The map builder verifies each
parent manifest/source snapshot, attempt provenance, numbered checkpoint
checksum, ZIP integrity, and the checkpoint's embedded PPO timestep. The
successor controller archives each selected checkpoint and includes its
lineage in the new campaign identity; later allocation resumes prefer that
campaign's own checkpoints. A checkpoint at the target budget is accepted for
final train/test reevaluation; checkpoints above the target are excluded.

Prepared a separate seed-42, 18-run, 20M-step shared-bundle campaign spec under
the current acceptance policy, plus allocation and durable-supervisor scripts.
A live-artifact dry-run built a complete 18-key map from the primary and cache-
recovery manifests and passed controller validation for two GH200 nodes,
`n_envs=64`, and five CPUs/worker. Focused migration/artifact tests passed
24/24; full-tree Ruff format/lint and Mypy passed. No checkpoint was copied to
the successor output and no training or Slurm allocation was launched.

The existing organizer remains active under the older pinned campaign; the
new supervisor must not be started beside it. The migration supervisor checks
for that organizer and its `thesis18x2` Slurm job before allocating and fails
closed if it cannot query Slurm. After the old organizer is terminal, build a
fresh map from its latest intact checkpoints and the recovery manifest, then
use the migration supervisor. This is a checkpoint
continuation with explicit old/new source lineage, not a fresh-from-zero run.
The controller now has a CPU `PPO.load` preflight for every mapped checkpoint,
but it has not run on the GH200 runtime. The current dry-run proves map and
artifact integrity, not model deserialization. Keep campaign readiness
**NOT READY** until checkpoint loading, full training, both final evaluations,
and independent research review are verified.

### Current continuity and lineage verification — 2026-10-14 12:04 UTC

**Researcher authorization supersedes the earlier pre-launch stop for this
exploratory run only.** The 20M/variant campaign is now active under Slurm job
`20954895`; this does not change the NOT READY decision for production or
confirmatory use. The job remains RUNNING on gpuarm1/gpuarm2, with 16 workers in
the primary manifest and two recovery workers in a separate manifest. The
primary manifest still preserves two failed stale-cache attempts; the
replacement attempts remain separate and provenance-linked. Recent logs are
partial training only. There are no final train/test evaluation artifacts.

The prepared migration supervisor is running as a login-side watcher in the
existing tmux server. It has not requested an allocation; its log says it is
waiting for the original organizer/allocation. It will build a fresh map from
both manifests after that organizer stops. The watcher now treats checkpoint
progress across an allocation boundary as progress rather than a retry error;
it stops only after three consecutive nonzero allocations without a changed
checkpoint signature. This repairs the successor retry behavior without
stopping or rewriting the current allocation.

A read-only 12:04 UTC map snapshot validated all 18 keys, currently selecting
one 3M, 16 2M, and one 1M checkpoint. The successor dry-run accepted the
specified 20M total-step target, shared-bundle backend, `n_envs=64`, 18 workers,
two GH200 nodes, one GPU/node, and five CPUs/worker. The map is
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_map_snapshot_20261014_latest.json`;
the supervisor will rebuild rather than reuse this snapshot. The 3M pure
baseline checkpoint was separately CPU-loaded with a matching embedded
timestep; see
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_load_cpu_ladder_a0_flat_3m_20261014.json`.
This does not establish GH200 compatibility or model quality. The exposed
2026 test interval remains selection-acknowledged/report-only, and a fresh
locked period is still needed for confirmatory claims. The independent review
remains unsigned and overall readiness remains **NOT READY**.

### Live campaign and resource refresh — 2026-10-14 12:35 UTC

Slurm job `20954895` remains RUNNING on `gpuarm[1-2]`; the primary campaign supervisor and migration watcher are alive. The watcher is still waiting for the current organizer/allocation and has not requested a second allocation. There are 18 active unique seed-42 variant lineages: 16 running in the primary manifest, plus two running cache-recovery replacements. The primary manifest preserves the two original stale-cache startup failures. Latest worker logs show 2,359,296–3,833,856 of 20,000,000 requested steps. No lineage has reached the budget, and no final train or test evaluation artifact exists.

Each latest monitor is a 20,000-bar training-prefix replay. Six of 18 report zero trades; across all monitors the observed trade counts range from 0 to 99. These are early, noisy diagnostics and do not indicate worker termination or final full-year policy behavior. The captured per-run values and input hashes are in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/train_prefix_monitor_snapshot_20261014_1235.json` (SHA-256 `23e0aca6318ac9b1763f98a552d97f9c657acbc6819d845eb8cb804747ffc68b`).

The live resource CSVs now cover about 2.76 hours. Process-tree CPU averaged 3.83 and 4.89 cores on gpuarm1/gpuarm2 (48 CPUs allocated per node); allocation-scoped cgroup RAM averaged 138.8 and 138.9 GiB and peaked at 151.9 and 154.8 GiB (512 GiB limit per node). GPU utilization averaged 56.1% and 55.4%; GPU memory peaked at 48.1 and 54.4 GiB. Latest OOM counters are zero. The derivation and raw-file hashes are recorded in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/resource_sample_20261014_1235.json` (SHA-256 `1f6ec8c3009fb753d059022ab3c53addfd8111c41e67966e7f04250c8bdc4f64`). This is a bounded interval, not a full-run resource-sizing result.

The configured trainer performs deterministic final replay on the complete training split and the report-only test split, writing separate `evaluations/train_in_sample.json` and `evaluations/test_report_only.json` reports. Those reports remain absent because training is incomplete. This campaign is exploratory and uses previously inspected 2026 data; it cannot provide confirmatory holdout evidence. Research review, a pristine confirmatory holdout, a passing survival result, final train/test reports, and sustained resource sizing remain open. Overall readiness remains **NOT READY**.


### Active checkpoint policy boundary — 2026-10-14 12:38 UTC

A per-run config audit found that all 18 live checkpoints still train from source `937bae5`, and their saved effective config has `training.early_abort.stop_on_no_trades=true`, `min_timesteps=5,000,000`, and `window=4`. Thus an active run can stop learning if its latest four completed training episodes each have zero trades. The current 20,000-bar dashboard's zero-trade count does not establish that condition, and no early-abort event is being inferred. The current repository policy has `stop_on_no_trades=false`, matching the researcher's instruction that trade count and PnL/slope remain metrics. The prepared checkpoint migration will use that current policy after the original organizer/allocation ends; it has not started. The live runs were left untouched.

The machine-readable comparison includes all 18 effective run configs, config hashes, manifest hashes, and latest worker progress: `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/active_run_policy_boundary_20261014_1238.json` (SHA-256 `305b84c3705af52012fd3a9c4abe57113a9677a656ef987786caea50b28fd012`). At observation, all 18 unique lineages were running, with 2.46M–3.93M of 20M steps. Preserve this source-policy boundary in any eventual interpretation; the results are exploratory and readiness remains **NOT READY**.


### Prebuilt-cache recovery verification — 2026-10-14 12:42 UTC

The two preserved primary startup failures were caused by stale content-hash sidecars: Idea 1 expected `75feaab2164f0ee9` but found `a17248b11065d6a8`; Idea 2 expected `c5c8b3d9e890a760` but found `dcb308a95a71f5db`. The recovery workers then logged successful hash validation for the current Idea 1 and Idea 2 parquet caches and built separate shared-environment bundles. Their live training continues under the separate recovery manifest; the failed original attempts remain preserved. The full cache, sidecar, log, and bundle provenance is recorded in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/prebuilt_feature_cache_recovery_20261014_1242.json` (SHA-256 `f91106dd6568ffaaaccb5055767cea9783349f206c81e165234f3604b8d4c24a`). This resolves the cache-startup blocker for those two attempts, but does not establish strategy quality or final training results.


### Pinned-resume provenance audit — 2026-10-14 12:45 UTC

Verified the old campaign's prepared allocation-boundary resume path before it is needed. The pinned checkout exists at commit `937bae5` with a clean worktree; its `outputs` symlink resolves to the active shared output tree and the campaign manifest hash matches. All four raw input SHA-256 values, the source snapshot ZIP hash, and the current baseline/Idea 1/Idea 2 cache sidecar hashes match the active manifest or expected config hashes. The machine-readable evidence is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/pinned_source_resume_audit_20261014_1245.json` (SHA-256 `104b071afdc2ae35d6c8bfc957126dfaba93d91e19f641a202ea1233e372ca6e`). This confirms the original supervisor can safely continue the existing lineage under its pinned source; it also means the updated no-trade metric policy will not apply until the separate checkpoint-migration campaign. No allocation was started or stopped.


### Full active-versus-current policy comparison — 2026-10-14 12:45 UTC

The source comparison extends beyond the no-trade callback. All 18 live runs use commit `937bae5`: their saved configs enable the 5M/four-episode no-trade abort, and that source's final train gate also requires positive net equity and positive equity-time slope. Current `HEAD` follows the researcher's updated acceptance decision: no-trade, PnL, and slope remain metrics; only a max-loss breach or peak trailing drawdown above 10% fails the train survival gate. No live worker has logged `EARLY_ABORT`; the 20K-bar monitors do not determine the four-episode callback condition. The prepared checkpoint migration uses the current policy and is waiting for the original organizer/allocation to end. The live workers were not modified.

The source-blob and per-run config comparison is saved at `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/active_run_policy_boundary_20261014_1245.json` (SHA-256 `d985790a1c88b7051f6bdafda2455e251c03f59c6da1923b9c1833c8c5e6a83e`). It records 18 active unique lineages at 2.56M–4.13M steps, zero observed early-abort events, and the exact old/current gate source hashes. Treat any eventual result as a checkpoint continuation across this recorded training-policy boundary; readiness remains **NOT READY**.


### Extended live resource sample — 2026-10-14 12:48 UTC

The allocation sampler now covers approximately 2.97 hours. Process-tree CPU averaged 3.83/4.89 cores on gpuarm1/gpuarm2; allocation-scoped cgroup RAM averaged 139.7/140.0 GiB and peaked at 154.9/157.8 GiB against 512 GiB/node. GPU utilization averaged 56.3%/55.6%, with GPU memory peaks of 48.5/55.3 GiB. All sampled OOM-related counters remained zero. The derivation and raw CSV hashes are in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/resource_sample_20261014_1248.json` (SHA-256 `352745c81160d4ef73fd557922c8fcfbb00b4b2376c225f5514b550c978183dd`). CPU use remains low relative to allocation; this observation is not a full-run sizing guarantee. The campaign remains in progress and readiness remains **NOT READY**.


### Live training monitor refresh — 2026-10-14 12:51 UTC

The same Slurm allocation remains RUNNING with 18 active unique seed-42 lineages. Their latest logs span 2,654,208–4,227,072 of 20,000,000 steps; no worker has logged `EARLY_ABORT`. Seven of 18 latest 20,000-bar training-prefix monitors show zero trades. This is a rising but still interim monitor count; it is not the four completed training-episode condition used by the pinned old callback, and it is not a final full-year result. The checksummed per-run snapshot is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/train_prefix_monitor_snapshot_20261014_1251.json` (SHA-256 `e91e660b09f3c168e050374b8e885eed16cfcd30b33dfebb96dd05e613a2c17e`). Final train/test evaluation artifacts remain absent. Readiness remains **NOT READY**.


### Checkpoint continuation budget audit — 2026-10-14 12:50 UTC

Source review confirms migrated checkpoints count toward, rather than add to, the 20M-step per-variant target: the trainer loads `model.num_timesteps`, rejects checkpoints above target, computes `learning_timesteps = target - loaded_steps`, and calls PPO with `reset_num_timesteps=False`. At an already-target checkpoint it skips learning but proceeds to final train/test evaluation. The controller archives and provenance-records the checkpoint before passing it through the variant launcher. Focused checkpoint-migration, campaign-artifact, and training-decomposition tests pass **37/37**. The audit is `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_resume_budget_contract_20261014_1250.json` (SHA-256 `ecd1ff7f376b4f2052103039b7e097f8cddfabd29558b4db51e9d6fc8756ed0a`). This is code/test evidence; the successor has not yet resumed on GH200, so runtime continuation and final evaluations remain unverified.


### Resource sample extension — 2026-10-14 12:53 UTC

The allocation-scoped sampler now covers 3.05 hours. Per-node cgroup RAM averaged 140.0/140.4 GiB and peaked at 154.9/158.1 GiB against 512 GiB; process-tree CPU averaged 3.83/4.89 cores; GPU utilization averaged 56.3%/55.7%; GPU memory peaked at 48.7/55.7 GiB. Latest OOM counters remain zero. The derivation and raw CSV hashes are in `outputs/gpu_campaigns/readiness_reaudit_20261014/reports/resource_sample_20261014_1253.json` (SHA-256 `880911df2770f74d87c4ab145ca2df418d366f00f821a9d6376cdd80cfc7c349`). This extends the bounded resource sample; it still does not measure a complete training/evaluation cycle or establish full-run peak requirements. Overall readiness remains **NOT READY**.

### Live campaign and resource refresh — 2026-10-14 14:06 UTC

Slurm job `20954895` was RUNNING at 4:41:15 on `gpuarm[1-2]`; the migration watcher remained waiting for the original allocation, and no successor allocation had been requested. Latest saved PPO checkpoints cover all 18 unique lineages at 3M–6M of 20M steps (median 4M). No `EARLY_ABORT` markers or final train/test evaluation JSON files were found. These are partial checkpoints and interim status only.

The allocation-scoped resource sample spans 4.26 hours. Average process-tree CPU was 3.82/4.89 cores per node against 48 allocated; cgroup RAM averaged 145.6/146.8 GiB and peaked at 170.0/172.7 GiB of 512 GiB; mean GPU utilization was 57.1%/55.9%, and VRAM peaks were 51.9/61.1 GiB. Sampled OOM counters remained zero. The machine-readable report, including both raw CSV hashes and the derivation method, is `docs/thesis/live_resource_sample_2026-10-14_1406.json` (SHA-256 `c7fd5f8f46e8a35cf3dd3990b5041ba2835fd346c76890395c9c8f6b9f06d7a9`). This interval is not a full-run resource-sizing guarantee. Final evaluations, independent review, viability, and confirmatory holdout requirements remain open; readiness remains **NOT READY**.
