# Independent Research Review Checklist — 2026-10-14

**Status: OPEN — no independent reviewer has completed or signed this review.**

This checklist prepares the remaining independent review of the Aalto thesis
trading-system evidence. It is not itself independent sign-off, a strategy
selection, or evidence that a strategy works. The internal source review is in
[`research_path_review_2026-10-13.md`](research_path_review_2026-10-13.md); the
campaign decision and evidence boundary are in
[`18_variant_training_readiness_plan.md`](../training/18_variant_training_readiness_plan.md).

## Evidence to review

- The 18-variant 100k exploratory results:
  [`campaign_results_summary.md`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md)
  and its JSON report. All 18 training equity gates failed. Twelve test
  replays failed the 130-day survival requirement; the remaining six made at
  most two trades. All 36 saved train/test PnL values reconcile with their
  equity curves. The 2026 test interval was previously inspected, so these
  results are exploratory and report-only.
- The 100k resource observation:
  [`resource summary`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md).
  It records 95 samples, 94 active, PSS peak 251.2 GiB, sampled GPU
  utilization peaks 97%/96%, and lacks allocation-wide cgroup memory and
  process-tree CPU-time measurements. It does not establish a sustained 20M
  resource profile.
- The current-source real-market feature-prefix audit:
  [`feature_prefix_causality.json`](../../outputs/gpu_campaigns/readiness_reaudit_20261014/feature_prefix_causality.json).
  It passed 9/9 comparisons across the three registered profiles and three
  cutoffs; recorded source, script and raw-data hashes were rechecked against
  the current tree. Its scope is prefix invariance at those cutoffs only.
- The current-source full-calendar frames/shared-bundle audit:
  [`frames_bundle_parity.json`](../../outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.json).
  It completed all 18 variants, comparing a 49,925-step exposed test-calendar
  trace and a 10,000-step active-action trace per variant, with zero failures
  and maximum scalar delta 0.0. Its 166 source/config hashes, four raw-data
  hashes and dirty-path manifest were rechecked against the current worktree.
  The result is bounded to those fixed traces; review it alongside the
  execution/reward source contract, not as proof of exhaustive policy-state
  equivalence or strategy performance. See the
  [human-readable summary](../../outputs/gpu_campaigns/readiness_reaudit_20261014/frames_bundle_parity.md)
  and [Slurm accounting](../../outputs/gpu_campaigns/readiness_reaudit_20261014/reports/slurm_accounting_20951231.psv).
- The [validation protocol](validation_protocol.md),
  [threats to validity](threats_to_validity.md), and current source review.

## Reviewer decisions requested

| Topic | Question for the independent reviewer | Current boundary |
| --- | --- | --- |
| Experiment design | Do the registered contrasts support the thesis claims? Which results, if any, are valid within each family? | Baseline, PO3/IFVG, and distribution families differ in reward and action/strategy semantics; do not interpret cross-family differences as isolated causal effects. |
| Execution and accounting | Are order timing, tick/quote fills, stops, take profits, costs, equity, and drawdown calculations correct for the stated market assumptions? | Code review and focused regressions exist; no independent broker integration or exhaustive reviewer sign-off exists. |
| Reward and risk | Are the training/evaluation breach penalties, shaped rewards, risk limits, and hard train-equity/survival gates appropriate and consistently interpreted? | Reward is not a profitability metric. All 18 historical 100k pilot gates failed under the then-current rule. The successor policy treats PnL, equity slope and activity as report metrics; max-loss breach and peak trailing drawdown above 10% are hard training-survival failures. The active 20M workers are pinned to the older policy, including its PnL/slope gates and no-trade early-abort; see the campaign status addendum. Review both policies and the final train/test reports without reinterpreting historical outcomes. |
| Feature causality | Are the MTF close-time alignments, swing/structure availability, train-only fitting, split boundaries, and cache identities causal? | On 2026-10-14, seven MTF/HTF/variant/swing/sweep/PO3/IFVG files passed 126/126 fixture tests in 5.76s; six causal-feature/split/fit-scope/cache/assembled-causality/leakage files passed 29/29 in 7.01s with two expected Transformer warnings; four entry-gate/eligibility/golden-trace files passed 30/30 in 3.50s (Python 3.12.14, pytest 9.1.1). Current real-market prefix invariance passed 3 profiles × 3 cutoffs with matching hashes. The current-source frames/shared-bundle audit passed 18/18 variants over the full 49,925-step test-calendar trace and 10,000-step active trace per variant; all 36 traces had maximum scalar delta 0.0, with matching source and raw-data hashes. Together these checks strengthen bounded feature/backend causality and parity evidence; they do not establish every execution/label/model-selection path or pristine confirmatory OOS. |
| Model selection | Is the selection protocol protected from exposed-period and human checkpoint-selection leakage? | The 2026 interval is permanently exposed. The one-seed campaign has no selected candidate and no independent selection record. |
| Performance claims | Do any results support viability, generalization, or confirmatory conclusions? | No. The 18-run pilot failed every training gate; six surviving test replays had at most two trades. No pristine holdout result exists. |
| Resource/reproducibility | Are the observed pilot resources sufficient for the proposed 20M campaign, and can the results be reproduced from retained inputs/source? | The latest approximately 1.98-hour window averaged 3.84/4.89 process-tree CPU cores and 135.9/134.9 GiB cgroup RAM, peaking at 152.9/147.8 GiB against 48 CPUs and 512 GiB per node; sampled GPU averages were 55.6%/55.5%, with zero OOM events. This is still an early-run sample, not a full-budget sizing result. At 12:01 UTC, job `20954895` was RUNNING; 16 primary workers and two separate recovery workers had recent logs between 1.9M and 3.1M steps, with no final evaluations. The campaign status JSON files were stale; Slurm and worker logs are the current evidence. Two initial stale-cache attempts remain preserved. The active workers pin source `937bae5`; current source is `bac4b4d`. The migration watcher is waiting for the old organizer/allocation and has not requested another job. Its checkpoint-progress retry fix is in `bac4b4d`; focused migration tests pass 5/5. Review both source manifests, migration lineage report, logs, and resource CSVs. Neither campaign's source identity or history should be edited. |

## Reviewer record

- Reviewer name and affiliation: ______________________________
- Relevant expertise / conflicts: ______________________________
- Materials and source revision reviewed: ______________________________
- Date: ______________________________
- Disposition: `ACCEPT` / `ACCEPT WITH CONDITIONS` / `REJECT` / `MORE EVIDENCE REQUIRED`
- Required changes or limitations to state in the thesis: ______________________________
- Signature or durable approval reference: ______________________________

Until a qualified reviewer fills this record and resolves the open items, the
independent-review gate remains **OPEN** and overall campaign readiness remains
**NOT READY**.

## Campaign and evaluation status addendum — 2026-10-14

The primary 18-run campaign is still active under its pinned source revision
`937bae529fdf5d6c17db6174880ef448320e2883`. At the latest status check,
Slurm job `20954895` was `RUNNING` on `gpuarm[1-2]`; the primary manifest had 16
running runs and two failed cache attempts, while the separate recovery
manifest had both replacement runs running. Their logs showed training progress
of roughly 1.18–1.87M timesteps in the primary campaign and 1.47M in each
recovery worker; no run had produced final train/test evaluation artifacts.
The old workers use the earlier acceptance rules, including the no-trade
early-abort and PnL/slope training gates. Do not reinterpret their outcome
under the newer survival-only acceptance rule.

The live resource CSVs cover about 78–83 minutes (142–150 samples). Allocation
cgroup high-water marks were 152.9 GiB on gpuarm1 and 142.1 GiB on gpuarm2,
against 512 GiB per node; current cgroup use was about 136.8/138.7 GiB, with
zero OOM events. The per-GPU utilization sample means were about 56% and 55%,
with observed peaks of 98% and 97%. Process-tree CPU time averaged about 3.86
and 4.89 cores for the primary campaign; the recovery campaign added about
1.09 cores on gpuarm1, for roughly 4.95 and 4.89 measured cores against 48
allocated cores per node. The CPU figures are calculated from each CSV's
cumulative process-tree CPU-time delta divided by its sample-window duration;
the cgroup figures cover the whole Slurm allocation. These early samples show
substantial CPU and RAM headroom and moderate GPU use, but cover only the early
training phase and do not establish full-run resource peaks or justify a
different `n_envs` setting by themselves. Source measurements are in the
primary and recovery `logs/resources_gpuarm*.csv` files.

At the same live check, `quota -s` reported 138 GiB of the user's 536 GiB
scratch quota used, with 456k of 1.048m files. This leaves roughly 398 GiB of
user quota headroom; the aggregate Lustre `df` capacity is not the applicable
per-user limit. The primary and recovery campaign directories used about
14 GiB and 100 MiB respectively; roughly 13 GiB of the primary total was the
parity cache. These are point-in-time storage figures, not a forecast of final
checkpoint/report growth or a Lustre I/O throughput benchmark.

A successor campaign has been prepared and pushed at repository commit
`2739387` (`Add provenance-safe campaign checkpoint migration`). Its seed-42,
18-variant, 20M-step specification retains the 2025 training and exposed 2026
report-only test split. The training entry point writes distinct
`train_in_sample.json` and `test_report_only.json` reports after training; the
same behavior is present in the current worker source pinned by the active
campaign. PnL, equity slope, and activity remain metrics; the updated hard
training-survival failures are a max-loss breach or peak trailing drawdown above
10%.

The successor controller's dry-run validated exact 18-key checkpoint lineage
from the primary and recovery manifests and the two-node, one-GH200-per-node,
48-CPU/512-GiB-per-node placement with `n_envs=64`. Full-tree Ruff format and
lint, full-tree Mypy (404 source files), and 35 focused bundle/migration tests
passed. The migration preflight now attempts strict reuse of the existing
primary campaign's content-keyed shared-environment bundles, while writing new
observation maps under successor output. It checks bundle identity and exact
array values and falls back to a fresh successor cache on any mismatch; it
does not repair or overwrite parent bundles. This could avoid rebuilding the
7.7 GiB shared-env portion of the current parity cache, but has not yet been
exercised on the compute runtime. The successor supervisor has **not** been
launched because the old organizer and allocation remain active. The
controller includes a CPU `PPO.load` preflight for each migrated checkpoint,
but no real migrated model
has yet passed that preflight on the compute runtime. This is tooling and
dry-run evidence, not completed training, final evaluation, or model-quality
evidence. The two campaigns and their source lineages must remain distinct.

The reuse decision is limited to the parent parity cache: all 18 variants and
36 traces in its report passed; all four configured raw-data hashes still
match; and all 167 old report source paths are present, with 163 hashes
unchanged. The four changed source files are limited to training early-abort,
survival-gate, and reporting behavior; the feature/data pipeline, environment,
bundle, worker, and parity-audit files are byte-identical. The successor audit
still rebuilds current features, compares them element-by-element with the
cached arrays, and runs both traces. A missing or mismatched cache triggers a
fresh successor-local bundle build. Actual compute-node reuse remains
unverified.

The active 2026 test calendar is previously exposed development data. Any
successor result on it is exploratory/report-only and cannot satisfy the fresh
locked-OOS requirement. The independent review record above remains unsigned;
the overall readiness decision remains **NOT READY**.

### Checkpoint-load audit update — 2026-10-14

All 18 currently available seed-42 checkpoints passed the CPU `PPO.load`
deserialization and embedded-timestep check (`18 passed, 0 failed`). Their
recorded checkpoints are 1M–2M steps. The report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_20261014.json`;
it records checkpoint hashes and lineage provenance. This is a bounded CPU
compatibility result. It does not verify loading on GH200, final-budget model
compatibility, continued training, completed train/test evaluation, or policy
quality. Slurm job `20954895` was still running at the latest status query;
the old campaign and recovery runs remain active and unchanged. Overall
readiness remains **NOT READY**.

### Live progress and continuity update — 2026-10-14 11:45 UTC

Slurm job `20954895` remains active on two GH200 nodes (48 CPUs and 512 GiB
per node, 6-hour limit). The primary campaign has 16 active workers and two
failed stale-cache attempts; two replacement runs are active in a separate
recovery campaign. Recent logs show roughly 1.6–2.7M and 2.0–2.1M steps,
respectively; neither manifest has final train/test evaluations. The saved
campaign status JSON timestamps are stale, so current state was checked from
Slurm, tmux, worker logs, and checkpoints.

The current organizer does not coordinate the recovery manifest and stops
after three allocation errors. A migration supervisor is now watching in the
existing tmux server and waits for the original organizer/allocation before
continuing all 18 checkpoint lineages under the prepared successor. It has
not started another allocation, and the live jobs remain untouched. The
worker source still uses the earlier acceptance policy; continuation under
the successor's current policy must retain that lineage boundary. No policy
quality or confirmatory evidence is available yet.

The latest approximately 1.98-hour resource window averaged 3.84/4.89 CPU
cores against 48 allocated per node, 55.6%/55.5% GPU utilization, and
135.9/134.9 GiB cgroup RAM against 512 GiB per node; peaks were 152.9/147.8
GiB with zero OOM events. These are early campaign measurements, not a
full-budget sizing guarantee. Preserve `n_envs=64` for the active and
checkpoint-continuation runs so PPO rollout/update behavior remains comparable.

The refreshed checkpoint audit loaded the newest available checkpoint from
all 18 seed-42 lineages on CPU: 18 passed, 0 failed; 13 were at 2M steps and
5 at 1M. The hash/provenance report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_latest_20261014.json`.
These were not final-budget models, and worker logs showed progress beyond the
saved checkpoint levels. Treat this as migration-input compatibility evidence
only.

The continuation supervisor now resets its allocation retry counter after new
checkpoint files appear and limits retries only after three consecutive
nonzero allocations with no checkpoint change. Shell syntax passed and the
focused checkpoint-migration tests passed 5/5; the watcher was restarted with
this change and is still waiting for the original allocation. This protects
continuity across allocation limits; it is not evidence that the strategy
learns or survives.

At 12:04 UTC, a fresh read-only successor dry-run validated 18 current
checkpoint lineages from the primary and recovery manifests: 1 checkpoint at
3M steps, 16 at 2M, and 1 at 1M. It accepted the intended 20M target, shared
bundle, `n_envs=64`, 18 workers, two GH200 nodes, and 5 CPUs per worker. The
snapshot map is
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_map_snapshot_20261014_latest.json`.
It is not the successor's launch map and will be superseded by a fresh map
when the waiting migration supervisor starts. This checks source lineage and
placement. The 3M baseline checkpoint separately passed CPU `PPO.load` with
its timestep matching 3,000,000; its hash/provenance report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_load_cpu_ladder_a0_flat_3m_20261014.json`.
This does not verify GH200 loading or model quality.

### Live exploratory campaign evidence refresh — 2026-10-14 12:26 UTC

Slurm job `20954895` is still RUNNING on `gpuarm[1-2]`; the original campaign
supervisor and migration watcher remain alive, with the watcher waiting and no
new allocation requested. The primary manifest records 16 running workers and
two preserved startup failures caused by prebuilt-cache hash mismatches. The
separate recovery manifest has those two run keys running, so there are 18
active unique variants across both manifests. Their latest worker logs show
2.26M–3.64M of 20M steps. A JSONL-based snapshot at 12:28 UTC found 8/18
latest 20,000-bar train-prefix monitors with zero trades; these are interim
training monitors, not final split evaluations or worker terminal states.
The per-variant metrics and source hashes are in
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/train_prefix_monitor_snapshot_20261014_1228.json`.
No final train-in-sample or report-only test report exists yet.

A freshly rebuilt lineage map covers all 18 variants from the primary plus
recovery campaigns. All 18 selected checkpoints passed local CPU
deserialization and embedded-timestep validation (one at 3M and 17 at 2M).
The report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_latest_20261014_1216.json`;
the map is
`outputs/gpu_campaigns/readiness_reaudit_20261014/checkpoint_map_snapshot_20261014_1216.json`.
The migration controller dry-run accepted those 18 lineages for the specified
20M, shared-bundle, `n_envs=64`, two-GH200-node configuration; its saved result
is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/campaign_migration_dry_run_20261014_1216.json`.
These results do not verify GH200 loading or final model quality.

The zero-trade train-prefix dashboard wording was clarified in commit
`c80a36c`; current workers remain pinned to source commit `937bae5`, so their
historical logs are preserved. The clearer label will apply to a later
migration run. The 2026 calendar remains previously exposed development data;
no fresh locked holdout is available in this scope. The independent reviewer
record remains unsigned. Do not infer survival, viability, or confirmatory
performance from these partial checkpoints and monitor outputs. Overall
readiness remains **NOT READY**.

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
