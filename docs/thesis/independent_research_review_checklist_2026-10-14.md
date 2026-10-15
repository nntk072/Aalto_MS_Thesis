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
- The [training-year coverage audit](train_year_coverage_audit_2026-10-14.json).
  It distinguishes 351,550 raw train-period M1 rows from 99,504 NY-session
  agent steps per complete environment episode. The 20M aggregate PPO budget
  at `n_envs=64` provides 312,500 nominal steps per environment, or about
  3.14 episode equivalents only when episodes reach the final training bar.
  Risk breaches end training episodes early, so this budget calculation does
  not prove that any policy completed a full-year training episode. Review the
  final train-in-sample and report-only test evaluations when available; they
  must cover their complete chronological splits.
- The `episode_equity.csv` files from any continuation using the current source.
  They append environment index, episode steps, full-split completion, and end
  reason after each episode. The original workers pinned to `937bae5` lack
  these fields, so checkpoint migration cannot retrospectively establish
  their episode lengths.
- The [validation protocol](validation_protocol.md),
  [threats to validity](threats_to_validity.md), and current source review.

## Reviewer decisions requested

| Topic | Question for the independent reviewer | Current boundary |
| --- | --- | --- |
| Experiment design | Do the registered contrasts support the thesis claims? Which results, if any, are valid within each family? | Baseline, PO3/IFVG, and distribution families differ in reward and action/strategy semantics; do not interpret cross-family differences as isolated causal effects. |
| Execution and accounting | Are order timing, tick/quote fills, stops, take profits, costs, equity, and drawdown calculations correct for the stated market assumptions? | Code review and focused regressions exist; no independent broker integration or exhaustive reviewer sign-off exists. |
| Reward and risk | Are the training/evaluation breach penalties, shaped rewards, risk limits, and hard train-equity/survival gates appropriate and consistently interpreted? | Reward is not a profitability metric. All 18 historical 100k pilot gates failed under the then-current rule. The active policy-migration run resumes checkpoints originating from source `937bae5` under campaign source `c2261ca`; this is a training-policy boundary that must remain explicit. Its effective configs disable no-trade and interim equity-gate early aborts. The current final training-survival gate treats PnL, equity slope, and activity as report metrics; max-loss breach or peak trailing drawdown above 10% fail survival. Review the old checkpoint phase, current continuation policy, and final train/test reports separately. |
| Feature causality | Are the MTF close-time alignments, swing/structure availability, train-only fitting, split boundaries, and cache identities causal? | On 2026-10-14, seven MTF/HTF/variant/swing/sweep/PO3/IFVG files passed 126/126 fixture tests in 5.76s; six causal-feature/split/fit-scope/cache/assembled-causality/leakage files passed 29/29 in 7.01s with two expected Transformer warnings; four entry-gate/eligibility/golden-trace files passed 30/30 in 3.50s (Python 3.12.14, pytest 9.1.1). Current real-market prefix invariance passed 3 profiles × 3 cutoffs with matching hashes. The current-source frames/shared-bundle audit passed 18/18 variants over the full 49,925-step test-calendar trace and 10,000-step active trace per variant; all 36 traces had maximum scalar delta 0.0, with matching source and raw-data hashes. Together these checks strengthen bounded feature/backend causality and parity evidence; they do not establish every execution/label/model-selection path or pristine confirmatory OOS. |
| Model selection | Is the selection protocol protected from exposed-period and human checkpoint-selection leakage? | The 2026 interval is permanently exposed. The one-seed campaign has no selected candidate and no independent selection record. |
| Training duration and evaluation | Does each policy receive a meaningful full-year training exposure, and do final evaluations cover the complete train and report-only test periods? | The nominal 20M/64 budget is about 3.14 NY-session episode equivalents per environment only if episodes reach the end. The latest immutable [`live campaign snapshot`](live_campaign_snapshot_2026-10-14_1950.json), SHA-256 `1f38c5eb626741a3221c96ebe879e147dc8369e1976c2b0eb34e6db3da9110af`, captures eight available episode telemetry files across eight variants: 225 completed episodes, all ending on trailing-drawdown breach at 7,338–71,857 of 99,504 train steps; none completed the split. Ten variants had no telemetry file at capture. Training resets return to the first eligible NY bar, so completed episodes restart on the same prefix; this does not prove what dates any still-running environment has visited or that every environment terminated. A tested full-split fix exists on unmerged branch `codex/full-year-episode-coverage` (commit `5f17bd5`); it is not active in the current campaign. No final train-in-sample or report-only test reports exist. The 18-worker run was still at 8.05M–12.82M of 20M when captured. Review per-environment coverage and complete-split reports before deciding whether the user's full-year learning requirement was met. |
| Performance claims | Do any results support viability, generalization, or confirmatory conclusions? | No. The 18-run pilot failed every training gate; six surviving test replays had at most two trades. No pristine holdout result exists. |
| Resource/reproducibility | Are the observed pilot resources sufficient for the proposed 20M campaign, and can the results be reproduced from retained inputs/source? | Current job `20963043` runs on two GH200 nodes, 48 CPUs and 512 GiB/node. The 19:49 UTC immutable 3.49-hour sample (`live_resource_snapshot_2026-10-14_1949.json`) measured mean cgroup RAM of 137.88/138.77 GiB and cgroup-reported peaks of 159.99/161.90 GiB, mean GPU utilization 58.7%/59.0%, peak sampled VRAM 65.87/99.26 GiB, and zero sampled OOM kills. Process-tree CPU averaged 4.92/4.89 cores per node; no CPU pressure/throttle counters were captured. Latest worker logs were at 8.05M–12.72M of 20M steps; no final train/test evaluation exists. This is partial resource evidence, not full-budget sizing. Campaign source is `c2261ca`; resumed checkpoint lineages preserve earlier source provenance `937bae5`. Review the campaign manifest, initial-checkpoint lineage, worker logs, and immutable raw resource samples together. |

The short three-trainer comparison is not a controlled benchmark: campaign `ladder3_shared_bundle_n64_seed42_20m_20261014` used `n_envs=64`, 32 CPUs/worker and two GH200 GPUs for three workers (1.5 workers/GPU); it logged 1,256–1,455 FPS/worker, or 4,017 aggregate steps/s. Current job `20963043` uses the same `n_envs` and GPU model but nine workers/GPU and five CPUs/worker; latest logs were 247–385 FPS/worker (median 319.5), or 5,486 aggregate steps/s. Aggregate throughput is higher by about 37%, while per-worker FPS is about fourfold lower. Source revisions and A0 `strategy_actions` differ, so don't treat this as a causal hardware benchmark. Each trainer launches 64 SB3 subprocess environments; the current resource sample counted about 615 processes per node. Early samples show CPU and RAM headroom and GPU peaks near 100% but means below 60%; sharing, process scheduling and rollout synchronization are plausible contributors, not an isolated proven bottleneck. No CPU-throttling counters were sampled. See the readiness plan's “Three-trainer versus 18-trainer throughput comparison” evidence and caveats.

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

### Live campaign and resource refresh — 2026-10-14 13:09 UTC

Slurm job `20954895` remains RUNNING on `gpuarm[1-2]` (2 GH200, 96 CPUs, 1 TiB
total RAM, six-hour allocation). Reconciliation across the primary manifest
(16 running, two stale-cache startup failures) and the separate cache-recovery
manifest (two running) confirms 18 active unique seed-42 lineages. Latest
logged progress is 2,949,120–4,620,288 of 20M requested steps. No early-abort
marker was found, but no final full-train or report-only test evaluation has
been produced. The current-source survival-only acceptance policy differs from
the pinned worker source `937bae5`; any result is an exploratory continuation
across that recorded policy boundary. The 2026 report-only period was already
inspected and remains selection-acknowledged, not pristine holdout evidence.

The allocation sampler covers 3.32 hours: process-tree CPU averaged
3.82/4.89 cores, cgroup RAM averaged 141.2/141.8 GiB and peaked at
158.7/162.6 GiB, GPU utilization averaged 56.5%/55.6%, and VRAM peaked at
50,499/58,175 MiB; OOM counters remain zero. This is interim sizing evidence.
The checksummed worker-level snapshot and raw resource CSV hashes are in
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/live_campaign_snapshot_20261014_1309.json`
(SHA-256 `af80c4cf4d7be47d9adb1a9caad263970559a619c9f51bf833e97f0e11075e07`).
The recovery controller log contains validation-error text even though the two
recovery manifests/logs show live, advancing workers; keep this invocation
discrepancy visible for review and do not treat the messages alone as proof of
failed workers. Independent review remains OPEN pending a signed reviewer
disposition, final train/test reports, a fresh confirmatory holdout, and
resolution of the policy-boundary and resource-sizing questions.

The latest selected checkpoint from each of the 18 lineages subsequently passed
CPU deserialization and saved-timestep validation (18/18; 3M–4M steps). The
report is
`outputs/gpu_campaigns/readiness_reaudit_20261014/reports/checkpoint_load_cpu_18_seed42_live_20261014_1316.json`
(SHA-256 `2ecff25d092504db675dcceb0ab38b08d996a81c31a63be89af471ca989fb9f7`).
At 13:28 UTC the same Slurm job remained RUNNING at 4:03/6:00, and worker logs
spanned 3.146M–5.112M steps. This does not verify GH200 loading or any final
train/test outcome; the reviewer gate remains OPEN.

At 13:48 UTC, a fresh audit of the latest resumable checkpoint for each of the
18 lineages passed provenance/hash checks and CPU deserialization (18/18;
3M–5M steps). The report is
`docs/thesis/checkpoint_load_cpu_18_seed42_live_20261014_1338.json` (SHA-256
`75581ab0b06107c232e458479673d57134dfcac167884a2f080cfdd31e41f1e3`). The
concurrent campaign remained RUNNING at 4:23/6:00, with six zero-trade
20,000-bar training-prefix monitors and no final train/test reports. This is
interim operational evidence only; independent review remains OPEN.

The 13:52 UTC resource refresh covers 4.02 hours: process-tree CPU averaged
3.82/4.89 cores on the 48-core nodes, cgroup RAM averaged 144.4/145.5 GiB and
peaked at 166.5/170.0 GiB, GPU utilization averaged 57.0%/55.9%, VRAM peaked
at 51.2/60.0 GiB, and sampled OOM counters were zero. The raw CSV hashes and
worker state are recorded in
`docs/thesis/live_campaign_snapshot_2026-10-14_1352.json` (SHA-256
`fc11afa4ac1779fb1179bbac452e369fa3fbfce262a536c010483d058082f209`). This
interim sample does not close full-run resource sizing or the review gate.

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

### Reporting change and live-state note — 2026-10-14 15:03 UTC

Current repository `main` is `a5e8e5f` (pushed). The campaign report now reads and hashes per-environment full-training-split episode telemetry, and the controller renders the JSON/Markdown summary after an allocation-boundary or terminal snapshot. The current `20954895` workers remain pinned to `937bae5`, so this feature applies to the distinct checkpoint-migration campaign and cannot reconstruct prior episodes.

At this observation Slurm job `20954895` was RUNNING at 5:38/6:00 on gpuarm[1-2]. The primary campaign manifest showed 16 workers running and two initial cache-sidecar failures; the separate cache-recovery manifest showed two replacement lineages running. The initial failures occurred before training because the Idea 1/Idea 2 cache sidecars differed from the effective expected hashes. Latest saved primary checkpoints reached 6M/20M; no complete train/test report was present. The migration watcher was still waiting for the original allocation. These are interim operational facts; they do not indicate viability or full-year episode completion. The external research reviewer has not signed this checklist, and the overall readiness decision remains **NOT READY**.

At 15:09 UTC, the allocation-scoped sample covered 5.32 hours: average RAM was 150.7/152.4 GiB and peaked at 180.6/184.0 GiB per node, average primary sampler process-tree CPU was 3.82/4.89 cores, average GPU utilization was 57.0%/56.4%, VRAM peaked at 54.6/65.9 GiB, and sampled OOM counters remained zero. The machine-readable snapshot is `live_resource_sample_2026-10-14_1508.json` (SHA-256 `a015994a3f2b176626b45f0bd66e5508166a73f490f246ff2f4c35d3818b50d4`); exact CSV snapshots and their hashes are linked there. This is not a complete campaign measurement. Job `20954895` remained RUNNING at 5:44/6:00, so the sample and worker status are interim.

### Live campaign and resource refresh — 2026-10-14 14:06 UTC

Slurm job `20954895` was RUNNING at 4:41:15 on `gpuarm[1-2]`; the migration watcher remained waiting for the original allocation, and no successor allocation had been requested. Latest saved PPO checkpoints cover all 18 unique lineages at 3M–6M of 20M steps (median 4M). No `EARLY_ABORT` markers or final train/test evaluation JSON files were found. These are partial checkpoints and interim status only.

The allocation-scoped resource sample spans 4.26 hours. Average process-tree CPU was 3.82/4.89 cores per node against 48 allocated; cgroup RAM averaged 145.6/146.8 GiB and peaked at 170.0/172.7 GiB of 512 GiB; mean GPU utilization was 57.1%/55.9%, and VRAM peaks were 51.9/61.1 GiB. Sampled OOM counters remained zero. The machine-readable report, including both raw CSV hashes and the derivation method, is `docs/thesis/live_resource_sample_2026-10-14_1406.json` (SHA-256 `c7fd5f8f46e8a35cf3dd3990b5041ba2835fd346c76890395c9c8f6b9f06d7a9`). This interval is not a full-run resource-sizing guarantee. Final evaluations, independent review, viability, and confirmatory holdout requirements remain open; readiness remains **NOT READY**.

### Interim train-prefix monitor refresh — 2026-10-14 14:12 UTC

A hashed snapshot of the latest 20,000-bar training-prefix monitor for each of the 18 unique seed-42 lineages reports seven zero-trade lineages. Across these interim monitors, trades range from 0 to 73, PnL from -$1,599.27 to +$2,650.36, and monitor timesteps from 3,801,088 to 6,201,344. The per-lineage records and source-file/record hashes are in `docs/thesis/live_train_prefix_monitor_2026-10-14_1412.json` (SHA-256 `2012a42f7e579b309f8fc55591ab2e2ea85126e4ee5b6caceb55230a180acdbd`). These are partial 20,000-bar diagnostics, not full-year train/test evaluations, survival determinations, or strategy-selection evidence; no final train/test report is present. Readiness remains **NOT READY**.

### Training-year budget and coverage clarification — 2026-10-14

A reproducible audit reconciles 351,550 raw training M1 rows with 99,504 eligible NY-session agent steps through 2025-12-31. At 20M aggregate PPO timesteps and 64 vector environments, each environment receives 312,500 nominal steps, or about 3.14 full-year episode equivalents only if episodes reach the final training bar. Risk breaches can end training episodes early, so this is budget capacity rather than proof of a completed full-year learning episode. Final deterministic train-in-sample and report-only test evaluations must cover their complete splits; neither report is present yet. The audit records distinct raw-row and eligible-session counting methods, source/config/log hashes, and environment step semantics in `docs/thesis/train_year_coverage_audit_2026-10-14.json` (SHA-256 `74f98db62dbcc89f90fca7876c705e1d52e79706208219fef45da0e0450a3a8f`).


### Active vector-resume campaign — 2026-10-14 16:56 UTC

The current continuation is campaign `ladder18_seed42_20m_policy_migration_nenv64_20261014`, Slurm job `20963043`, 18 seed-42 PPO variants, 20M requested steps each, shared bundle, `n_envs=64`, two GH200 nodes with 48 CPUs and 512 GiB RAM per node. The machine-readable live snapshot is [`live_campaign_snapshot_2026-10-14_1656.json`](live_campaign_snapshot_2026-10-14_1656.json), SHA-256 `a5f98d2c20835caca79d58fa1398d7b90c7027f7f0f7f6ac0aa6c998446569e4`. At capture all workers were running at 5.49M–8.79M logged PPO steps, with no error markers, no new numbered checkpoint yet, and no final evaluation artifacts.

The effective config audit is `outputs/gpu_campaigns/ladder18_seed42_20m_policy_migration_nenv64_20261014/reports/effective_config_audit_18.json`. It confirms no trade-count or equity-gate early abort, an unset artificial episode-length cap, and 1M-step periodic checkpoint frequency. The current-source evaluation contract audit verifies deterministic evaluation on the complete training split and report-only test split; it is static evidence only. Require the eventual `episode_equity.csv` and full `evaluations/` outputs before judging full-year training coverage or survival.

Checkpoint lineage begins from source `937bae5` and resumes under `c2261ca`; review this boundary when assessing training-policy continuity. The interim PPO numeric diagnostic records an extreme finite pre-clip gradient norm in `ladder_a4a_entry_fsm_hidden_dist`; default PPO clipping is 0.5. Treat this as a numerical warning for review, not a standalone failure or success claim. The 2026 period remains exposed and selection-acknowledged; it cannot support confirmatory OOS. Historical 100k pilot gates all failed. No candidate has final train/test results, independent sign-off remains unsigned, and overall readiness remains **NOT READY**.

### Live episode-telemetry refresh — 2026-10-14 17:06 UTC

The active job `20963043` was still running with all 18 manifest workers marked running and all 18 worker logs fresh within five minutes. One post-resume numbered checkpoint (`ladder_a0_flat__seed42`, 9M steps) is now present. Final train/test metrics and the final campaign summary remain absent.

Available episode telemetry currently covers only `ladder_a1_sl__seed42`: nine completed episodes, each ending in a trailing-drawdown risk breach after 7,338–13,630 steps, with 89–169 trades per episode. None reached the complete train split. This is a relevant interim survival signal, but it is not the final deterministic train evaluation or a full-campaign conclusion. The raw evidence and checkpoint lineage are recorded in [`live_campaign_snapshot_2026-10-14_1706.json`](live_campaign_snapshot_2026-10-14_1706.json), SHA-256 `589278652a60337dabe1c5b34400d15310a3806f2dc460cd44345fb9b26a92a4`.

The independent reviewer should examine these episodes alongside the eventual full-split train/test reports and the historical 100k failures; do not treat the interim episodes as independent confirmation or a selected strategy. Review remains unsigned and readiness remains **NOT READY**.

### Additional episode telemetry — 2026-10-14 17:07 UTC

The immutable 17:07 snapshot ([JSON](live_campaign_snapshot_2026-10-14_1707.json), SHA-256 `eb757f5256841f2cc85b14b6bc5775acd4d0f9c1c1d6269b99f2de4185b60500`) archives two current-source episode CSVs. `ladder_a1_sl__seed42` has nine recorded early episodes; `ladder_a1_sl_po3__seed42` has one. All ten ended on trailing-drawdown risk breach without completing the training split. The single Idea 1 episode lasted 14,731 steps and recorded 69 trades. These data warrant examination with eventual full-split evaluations but do not show final model performance. The active campaign has not produced final metrics, independent review remains unsigned, and readiness remains **NOT READY**.

### Episode-telemetry refresh — 2026-10-14 17:09 UTC

The immutable [17:09 campaign snapshot](live_campaign_snapshot_2026-10-14_1709.json), SHA-256 `11132512e5f0e3a5048e0b53fee94552a205794a9d3898f016a631ac99c8cfa8`, archives both currently available episode CSVs. Ten episodes from `ladder_a1_sl` and one from `ladder_a1_sl_po3` all ended on trailing-drawdown breach before the full training split completed. This is a concerning interim survival signal that the independent reviewer should compare with final deterministic train and report-only test evaluations. It is not final policy evidence: all 18 workers remain active, and no final metrics or summary exist. Independent review remains unsigned and readiness remains **NOT READY**.

### Live resource and episode evidence — 2026-10-14 17:11 UTC

The 50.5-minute allocation-scoped sample averages 124.12/122.70 GiB cgroup RAM per node, with recorded peaks of 132.14/132.80 GiB out of 512 GiB; process-tree CPU averages 4.939/4.904 cores; GPU utilization averages 55.2%/54.9% and peaks at 98%; GPU-memory peaks are 60,951/101,645 MiB; sampled OOM events are zero. This remains partial live-run evidence, not final sizing. The machine-readable report is `docs/thesis/live_resource_snapshot_2026-10-14_1710.json` (SHA-256 `b15fa89bce224ac65e063100835ace24fc22d465a2ca6a7007993befcccbc31c)` with immutable raw CSV copies and hashes.

The 17:11 campaign snapshot (`docs/thesis/live_campaign_snapshot_2026-10-14_1711.json`, SHA-256 `50b2558b0c6e27e01107d00ab9245afae3c696f4a81b6acb809c2f8d84ee8bad`) records 18/18 workers active and fresh logs; three numbered continuation checkpoints at 7M–9M; twelve episode records across three variants, all ending in trailing-drawdown breach before completing the train split; and no final metrics or summaries. These data are exploratory progress signals, not final results. Readiness remains **NOT READY**. The independent research-review record is still unsigned; examine this partial evidence with the final train/test artifacts when they appear.

### Reviewer evidence update — 2026-10-14 17:13 UTC

### Active 18-run campaign evidence — 2026-10-14 17:13 UTC

The immutable campaign snapshot [`live_campaign_snapshot_2026-10-14_1713.json`](live_campaign_snapshot_2026-10-14_1713.json), SHA-256 `c9df47a35c5ed838f31969385c9b49b723e3bf99a623ee614ddaa48dd2a3ef12`, records job `20963043` running at 1:09:45/6:00:00, 18/18 workers active, all worker logs fresh, and eight numbered continuation checkpoints. Current dashboard steps range from 5.70M to 9.18M of 20M total; models resumed from 5M–8M checkpoints. Mean added progress is about 0.90M per worker over 54 minutes (about 276 steps/s); a linear projection leaves about 13.4 hours to the 20M target and is not a runtime guarantee. The continuation supervisor is configured to request successive allocations when needed.

The snapshot preserves 14 completed environment episodes across three variants; all ended in trailing-drawdown risk breach, and none completed the full training split. No final metrics or final summaries exist. These are interim training alarms, not the final deterministic train/test evaluations or evidence that the finished policies fail or succeed.

The paired partial resource report [`live_resource_snapshot_2026-10-14_1713.json`](live_resource_snapshot_2026-10-14_1713.json), SHA-256 `7c81adfb211cd90c4f6ed0f7b00800708910d02ad5d8dd56c68c6cf5aab803fc`, covers about 53.6 minutes: mean job-cgroup RAM was 123–124 GiB per 512-GiB node, recorded peaks 132–134 GiB; process-tree CPU averaged about 4.9 cores/node while 45 worker CPUs were requested; GPU utilization averaged 55–56% and peaked at 98–99%; GPU-memory peaks were 61,065 and 101,645 MiB; sampled OOM counters were zero. This is partial resource sizing, not a full-run profile. Readiness remains **NOT READY**.
The independent reviewer should assess the repeated early trailing-drawdown episodes against the eventual full-split evaluations; this live telemetry does not substitute for the unsigned review.

### Requested CI checks and live campaign refresh — 2026-10-14 17:16 UTC

The default `.venv` could not run on this x86 login host (`Exec format error`); rerunning through the repository's `.venv-x86` succeeded: `ruff format --check .` reported 406 files formatted; `ruff check .` passed; `mypy .` reported no issues in 406 source files; and focused campaign/report/resource tests (`tests/test_triton_resource_sampler.py`, `tests/test_campaign_results_summary.py`, `tests/test_campaign_artifacts.py`) passed 31/31. `git diff --check` passed. No training-source code changed.

The newest point-in-time snapshot is `docs/thesis/live_campaign_snapshot_2026-10-14_1715.json` (SHA-256 `d3d74060630193779b6302dab79f7d4388d751185e2d1fd2b8990dfa4d58eb12`). Job `20963043` was still RUNNING at 1:11:51/6:00:00 with all 18 workers active and fresh logs. Ten numbered checkpoints existed. Fifteen recorded episodes across `ladder_a0_flat_dist`, `ladder_a1_sl`, and `ladder_a1_sl_po3` had all ended on trailing-drawdown breach before the full train split; no final metrics or summary existed. This remains interim evidence, not the final policy decision.

Full-tree format/lint/type and scoped tests are complete for the current docs/evidence work. The requested commit/push remains deferred until the campaign and continuation watcher are terminal: the resume fingerprint includes both Git SHA and source-file hashes, so a docs-only commit would change the fingerprint even though worker source hashes still match. Only docs/report artifacts are modified. Do not claim readiness: independent review remains unsigned, the campaign is incomplete, and exposed data remains exploratory. Overall readiness remains **NOT READY**.

### Live campaign refresh — 2026-10-14 19:39 UTC

Job `20963043` remains RUNNING at 3:35:47/6:00:00. Eighteen worker logs are fresh at 7.85M–12.52M/20M steps; no final train/test report exists. The slowest logged worker is near 248 FPS, roughly 13.61 hours remaining at a constant rate. The full-split training fix is pushed but isolated at commit `5f17bd5`; it is not active in this run. The 17:45 episode audit captured 28 episodes, none reaching the train-split boundary. The continuation watcher waits for the existing allocation and then two idle GH200 nodes before resuming the same fingerprinted campaign. These remain exploratory interim observations; independent sign-off, full-year training exposure, final train/test reports, viability and fresh confirmatory OOS remain open.

### Live campaign refresh — 2026-10-14 17:19 UTC

The point-in-time [campaign snapshot](live_campaign_snapshot_2026-10-14_1719.json), SHA-256 `d0a3fa93bf2382fdd3f4a8f25f5b114523112bf425feb6fb5b3ab1558db3cfee`, records job `20963043` running at 1:15:42/6:00:00, all 18 workers active with fresh logs, 13 numbered checkpoints, and latest dashboard steps from 5.80M to 9.30M of the 20M target. No final metrics or summaries exist.

Nineteen completed environment episodes are currently recorded across three variants; all nineteen ended on trailing-drawdown breach before completing the training split. This is an interim survival signal, not a final evaluation. The paired [resource snapshot](live_resource_snapshot_2026-10-14_1719.json), SHA-256 `a2ddc3fddc59a9c9f39d73074f9c9d4df2509d94a1e2524937a2a888320615cd`, covers about 60 minutes: cgroup RAM averages 123.9–124.8 GiB/node and peaks at 132–134 GiB of 512 GiB; process-tree CPU averages 4.90–4.94 cores/node; GPU utilization averages 55–57% and peaks at 98–99%; VRAM peaks at 61,315 and 101,645 MiB; sampled OOM counts remain zero. This is still a partial run profile. The source fingerprint remains unchanged, and the run is left active under the existing continuation watcher. Readiness remains **NOT READY**.

### Episode-outcome interpretation — 2026-10-14 17:20 UTC

Using the immutable episode CSV copies listed in `docs/thesis/live_campaign_snapshot_2026-10-14_1719.json` (snapshot SHA-256 `d0a3fa93bf2382fdd3f4a8f25f5b114523112bf425feb6fb5b3ab1558db3cfee`), I checked the actual end reasons and account deltas. All 19 recorded episodes across `ladder_a0_flat_dist`, `ladder_a1_sl`, and `ladder_a1_sl_po3` ended specifically on `trailing_dd`; none reached the 99,504-step full training split. Variant means were: A0 distribution, 14,611 steps / 61.7 trades / −$6,765 equity delta (3 episodes); A1 stop-loss, 13,059 steps / 157.5 trades / −$5,977 (15 episodes); A1 stop-loss plus PO3, 14,731 steps / 69 trades / −$6,732 (one episode). The longest recorded episode covered only 14.8–17.6% of the complete train split, depending on variant.

This rules out “no activity” as the explanation for these specific episodes: they traded, then hit a trailing-drawdown terminal. It is serious interim negative survival evidence, but the agents are still training and these are not final deterministic train/test evaluations. Keep the run’s eventual gates and complete-split reports as the decision evidence; do not label these partial episodes a final campaign failure or a successful strategy.

### Trailing-drawdown terminal adjudication — 2026-10-14 17:23 UTC

A source/config check confirms the observed episode endings are consistent with the configured training guardrail, not evidence of an accidental episode-length cap. All 18 current effective `config.yaml` files set `ftmo.trailing_dd_limit=0.07` and `env.max_episode_steps=null`. For example, A1-SL's effective config is `outputs/gpu_campaigns/ladder18_seed42_20m_policy_migration_nenv64_20261014/runs/attempts/ladder_a1_sl__seed42/attempt1/ladder_a1_sl__seed42/20261014_191951_745461_rl_train_seed42_tcn/config.yaml` (SHA-256 `c36f0f45246c5b417a710c8647fb69d3c99ede8878c9590fd13f5e78a14c0b2b`). `FTMOGuardrails.check_trailing_dd()` ends PPO training episodes when the peak-to-current equity drop reaches 7% of initial balance ($7,000 on $100,000); `TradingEnv._check_guardrails()` terminates episodic training on that breach. Non-episodic evaluation latches the year failure and continues the rollout. The focused `test_trailing_dd_terminates_year_episode` regression reports a pass.

Therefore the episodes' end equity delta from starting balance must not be compared directly to the $7,000 peak-to-trough threshold: prior gains can make peak equity exceed starting balance. The repeated breaches are genuine training-sample survival failures under the configured gate, not a code defect established by current evidence. They remain interim while the policies train; require the eventual full-split train/test reports before making a final policy decision.


### A0 reference-run contract check — 2026-10-14 17:51 UTC

Current runtime A0 config, cache profile, action semantics, and checkpoint provenance were reconciled in [`a0_current_action_feature_audit_2026-10-14.json`](a0_current_action_feature_audit_2026-10-14.json), SHA-256 `2637a60b14496d12f7f29802d16043addfa8834da696e9b794b2ec6e9bdf7887`. The active A0 is the intended pure legacy reference: `strategy_actions=false`, default features/cache, `Discrete(20)`, and no PO3/IFVG/FVG/manipulation feature columns; the PO3 action-side lock and strategy-action open-blackout filter are bypassed. This is a bounded config/source/runtime-log audit, not independent sign-off. A separate current-source A0 frames/shared-bundle parity audit now passes the fixed calendar and active-action traces; review its stated limits below. Overall review and readiness remain open / **NOT READY**.


### Resource sizing evidence refresh — 2026-10-14 17:55 UTC

The immutable [`live resource sample`](live_resource_snapshot_2026-10-14_1755.json), SHA-256 `0dd5a117a746397072b30e78ee6bedcd9a2cc9289bcf841d9c4d8c76c54aace4`, covers about 1.58 hours of the active six-hour allocation. Cgroup RAM averages about 128 GiB and peaked at 138.6/140.2 GiB on the 512-GiB nodes, with sampled OOM kills at zero. GPU averages were 58.5%/57.1%, with peaks at 99%/98%; VRAM peaks were 61.3/99.3 GiB. Slurm step AveCPU indicates about 4.9 CPU cores used across nine trainers requesting 45 cores per node. Treat this as a bounded early sample; full-run sizing and sustained aggregate throughput remain open.


### Corrected A0 backend parity evidence — 2026-10-14 21:03 Helsinki time

The current-source A0-only frames/shared-bundle audit passed with zero scalar delta across the 49,925-step test-calendar trace and 10,000 active-action steps; it opened 46 trades in the active trace. The report [`a0_frames_bundle_parity.json`](../../outputs/gpu_campaigns/readiness_reaudit_20261014/a0_current_parity/a0_frames_bundle_parity.json) has SHA-256 `6d40de0e4429ed2db0ce7016b0583d8362374f40a373834cea5df793c0921f86`; its retained driver hash is `81897d053d5ef7ee4190c8bb9e7e4421d6842dc63267d413f66e0b24527c094e`. This strengthens only bounded A0 backend parity. It does not satisfy independent review or prove strategy performance, and other current-source variant parity remains unverified.
