# Triton RAM optimization implementation plan
Date: 2026-10-10
Repository: /scratch/work/nguyenl37/Aalto_MS_Thesis
Scope: 18 variants x 5 seeds x 20M PPO timesteps.

## Verified evidence
- Default env.data_source=frames in quant_rl/config/default.yaml.
- train_rl.py already implements env.data_source=bundle, make_bundle_env, and _publish_obs_memmap.
- shared_bundle.py loads arrays read-only with np.load(mmap_mode="r"); TradingEnv consumes _bundle_arrays.
- tests/test_envs/test_shared_bundle.py covers atomic creation/concurrent builders.
- The existing GH200 campaign core-18x5 uses 2 trainers x 8 envs, allocated 32 CPU / 384 GiB; preserve its current state.
- Historic worker PSS at n_envs 8/16/32/64: 18.5/34.3/65.9/129.2 GiB (not cgroup RAM).

## Strict invariants
Never change features, observation/action definitions, rewards, execution prices, costs, SL/TP, exits, position sizing, guardrails, train/OOS split, purge, episode boundaries, metrics, or seed schedules. Keep exact financial precision; float32 downcasting requires explicit parity review. No training on login/Windows. No implicit new srun/scancel. No modifications to in-flight training runs.

## Stage 0: baseline attribution
Record git/config hash, Slurm allocation, actual CPU affinity, cgroup memory.max/current/peak/events, process USS/PSS/smaps_rollup, anonymous vs file-backed pages, disk free and scratch I/O. Collect 30-60-second time series and process trees while feature building, env creation, rollout, PPO optimizer, checkpoint, IS and OOS run. Separate node free memory from job cgroup memory and avoid adding shared RSS. Identify top 3 duplicated immutable buffers and estimate potential savings. Store raw observations as CSV/JSON under outputs/gpu_campaigns/<campaign>/logs/.

## Stage 1: audit existing shared bundle
Inspect env_bundle_builder.py, shared_bundle.py, env_spec.py, worker.py, TradingEnv.from_bundle, train_rl.py and SubprocVecEnv creation. Verify cache fingerprints, atomic build, concurrent writes, checksum/dtype/shape, split isolation, file-backed mappings and worker cleanup. Audit any array conversion, .copy(), DataFrame reconstruction, tickbook, OHLC, session and MTF arrays. Resolve all 18 variants, action widths, three strategy families and VAE/walk-forward compatibility. Unsupported cases must be explicit; do not silently change behavior.

## Stage 2: scientific parity, required before enabling
Build frames and bundle from the same synthetic and representative real fixtures. Replay same seeded action sequences over 1, 8, 16 environments, including no-context holds, entry signals, direction control, SL/TP and partial exits, trailing, FSM, tick costs, overnight, risk breaches, holidays, NaNs, and MTF boundaries. At every transition compare observations, reward, NAV, positions, executions, trade ledger, masks, entry diagnostics, done/truncated and cause. Require exact discrete decisions and specified tight numeric tolerances for floating quantities, with no systematic drift. Verify split and train/OOS cache separation, deterministic episode resets and resolved config equivalence. Extend bundle tests with real train/eval integration cases. Gate on passing ruff check, ruff format --check, mypy, pytest. Do not deploy on parity failure.

## Stage 3: measured optimizations only
A. Activate immutable bundle as a per-run opt-in; pass EnvSpec/path to workers, not large parent frames.
B. Remove extra copies of immutable feature, tick, OHLC and observation arrays only when profiling confirms them.
C. Release parent and child temporary pandas/intermediate arrays after mmap publication; preserve cache lifecycle.
D. Limit BLAS/OpenMP/PyTorch CPU threads where benchmarks show oversubscription. Never fork after CUDA initialization.
E. Tune storage layout, mapped pages, IPC, and feature cache only after measuring their share of memory/time.
F. Handle VAE latent-table separately with a fresh parity gate.
Never shrink scientific inputs, suppress evaluation, or silently change PPO n_steps/batch/epochs.

## Stage 4: true PPO benchmark
Profiles: workers x n_envs = 1x8, 1x16, 1x32, 1x64, 2x8, 2x16, 2x32; optionally 3x8, 4x8 after admission checks. Measure both frames and bundle at identical algorithm settings. Require multiple meaningful rollout + optimizer cycles per profile and repeated sustained samples where possible. Report separately feature preprocessing, worker startup, environment throughput, PPO optimization, checkpoint, IS/OOS time; full pipeline wallclock must not masquerade as PPO steps/s. Capture actual model timesteps (rollout rounding), GPU utilization/VRAM, allocated CPU utilization, cgroup peak/anon/file, USS/PSS, I/O and failure conditions. Admit only if >=64 GiB reserve plus 1.35x measured predicted peak fits allocated cgroup RAM; ensure CPU headroom. Select maximum aggregate trained timesteps/hour, not highest env count.

## Stage 5: safe pilot and rollout
Pilot a separately named campaign (baseline, PO3/IFVG, distribution, varied action widths) with seed 42 and validated checkpoints/metrics/trade gates. Keep existing core-18x5 logs/checkpoints untouched. Record git SHA, config digest, feature/cache hashes, backend, GPU/CPU/cgroup, seed, n_envs, PPO parameters, actual timesteps and artifact paths. Fix campaign-controller status to verify checkpoints + result outputs rather than exit code alone; preserve append-only attempt logs and resume without double scheduling. Switch production backend only at a safe run boundary, after CI + parity + performance gates. Check remaining allocation walltime, scratch quota, Slurm restrictions and availability before later jobs. Commit independently reviewable verified changes; do not claim hypothetical memory gains as measured.

## Acceptance
- All 18 variants resolve correctly and unsupported cases are flagged.
- Complete observational, financial, termination, train/OOS and action parity.
- Ruff format/lint, mypy and targeted tests pass.
- Lower cgroup peak memory for matched workloads (not merely lower RSS) and validated aggregate PPO training throughput.
- No new OOM, reproducible checkpoints, durable logs, resume and auditable scientific provenance.


## GPU bundle probe evidence (2026-10-10; exploratory)
All measurements ran on allocated GH200 GPU compute node in Slurm 20877341; existing prebuilt immutable bundles, CUDA PPO with 15 epochs, n_steps=2048, batch_size=512, and short runs. Saved raw JSON at outputs/gpu_campaigns/core-18x5/reports. These are NOT production throughput or peak RAM comparisons against frames and cannot authorize full rollout.

| Bundle | envs | steps | elapsed seconds | steps/sec | cgroup post-train GB decimal |
|---|---:|---:|---:|---:|---:|
| 26.8 MB | 8 | 4096 | 9.34 | 438.5 | 9.87 |
| 26.8 MB | 16 | 4096 | 5.82 | 703.2 | 10.63 |
| 26.8 MB | 32 | 4096 | 4.60 | 890.5 | 12.16 |
| 26.8 MB | 64 | 4096 | 3.55 | 1154.1 | 15.25 |
| 861.6 MB | 32 | 8192 | 9.01 | 908.7 | 12.24 |

Cgroup values are point-in-time, not monitored peaks. Probe memory includes baseline allocation. Workers scale memory even with mmap, and increased n_envs can cause CPU contention. Compare matched production configurations and collect peaks before choosing workers/parallelism. Hold campaign core-18x5 suspended pending full parity and workload comparisons.

## Matched seven-day frames/bundle memory comparison (2026-10-10)
GPU-node Slurm 20877341, same feature pipeline and seven-day train slice, environment-only vector step probe (128 steps), no PPO gradient phases. Full raw JSON is at outputs/gpu_campaigns/core-18x5/reports/{frames,bundle}_memory_matched.json (ignored run outputs).

| Backend | n_envs | cgroup start GB | cgroup steady GB | added cgroup GB | tree PSS MiB | startup s |
|---|---:|---:|---:|---:|---:|---:|
| frames | 8 | 10.63 | 13.84 | 3.20 | 6873 | 13.78 |
| bundle | 8 | 13.25 | 14.37 | 1.12 | 5886 | 1.94 |
| frames | 16 | 10.54 | 17.14 | 6.60 | 10011 | 26.61 |
| bundle | 16 | 13.62 | 15.14 | 1.51 | 6590 | 0.66 |

The different starting memory indicates cache/cgroup effects; incremental cgroup growth is more informative than absolute steady memory here, but not a peak. Bundles reduced incremental memory in this matched short-slice experiment by about 2.08 GB (8 envs) and 5.09 GB (16 envs), and reduced process PSS. Results do not extrapolate to 90 production runs. Next gates remain full train-split parity with all strategies, peak profiling of concurrent PPO gradient phases, and small end-to-end pilot with checkpoint and OOS validation. Do not restart core-18x5 on these findings alone.
