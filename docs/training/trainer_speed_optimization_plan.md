# Trainer Speed Optimization Plan

> **Plan routing:** Current order and next action are governed by the [master project execution plan](../thesis/master_project_execution_plan.md). This document is the technical measurement reference; do not execute it as a separate queue.

**Status:** measurement and design plan; no training source or active campaign configuration has been changed.  
**Context date:** 2026-10-14.  
**Purpose:** increase useful campaign throughput without weakening the training/evaluation contract or misreading per-worker FPS as total campaign speed.

## Current evidence

The active exploratory campaign runs 18 workers with the shared-bundle backend, `n_envs=64`, PPO `n_steps=8192`, `batch_size=4096`, and `n_epochs=15`. Each worker uses a 64-process `SubprocVecEnv`. Since the configured rollout size is divided across 64 environments, each environment contributes 128 steps per rollout; the rollout buffer contains 8,192 transitions. Each PPO update performs 30 minibatch updates (2 batches per epoch × 15 epochs).

The 2026-10-14 19:40 UTC comparison showed 1,256–1,455 FPS per worker and 4,017 aggregate steps/s in a prior three-worker run, versus 247–385 FPS per worker and 5,486 aggregate steps/s in the current 18-worker run. This is not a controlled comparison: source revisions and baseline action settings differed. It does show why campaign throughput must be judged by aggregate steps/s and completion time, alongside per-worker speed.

The 19:49 UTC resource sample for the current two-node allocation measured about 4.9 CPU cores of process-tree use per node, 137.9/138.8 GiB average cgroup RAM and 160.0/161.9 GiB cgroup peak out of 512 GiB, 58.7%/59.0% average GPU utilization with sampled peaks of 99%/100%, and no sampled OOM kills. CPU pressure/throttling counters were not collected. These measurements do not identify one proven bottleneck. They argue against RAM exhaustion or system-wide CPU saturation during this sample; worker scheduling, interprocess communication, rollout synchronization, GPU sharing, and callback work remain hypotheses to measure.

The training code already limits common numerical-library thread counts to one in environment workers. Raising CPU requests or memory alone therefore has no demonstrated path to restore the prior per-worker FPS. The existing benchmark script also does not match this campaign's PPO configuration, so its result must not be used for tuning until corrected.

## Constraints while the campaign is live

- Keep the active campaign's source fingerprint unchanged until the campaign and its existing continuation watcher are terminal. The resume fingerprint includes the Git SHA as well as source hashes; even a docs-only commit can invalidate automatic resume.
- Do not launch a second benchmark allocation while this campaign or watcher owns the available allocation path. Do not cancel existing Slurm jobs.
- Preserve the current run as exploratory evidence. Any speed-related change that affects policy updates, observation semantics, evaluation RNG, or episode exposure creates a new training lineage and requires a fresh, explicitly recorded run.
- Keep the full training and testing evaluation, survival diagnostics, and existing acceptance gates. Faster execution cannot come from silently omitting them.

## Staged plan

### 1. Establish an exact, phase-level baseline

After the active campaign and watcher finish, use an existing authorized GPU allocation or request one according to the Triton rules. First record the resolved config, Git SHA, data/bundle hashes, device, versions, and seed. Run one worker long enough to exclude startup and cache warmup, then separately time:

1. environment reset and observation construction;
2. rollout collection, including vector-worker waiting time;
3. policy optimization;
4. dashboard and evaluation callbacks;
5. checkpoint and report writing.

Report both steady-state steps/s and end-to-end elapsed time. Capture per-process CPU, cgroup CPU pressure/throttling, GPU utilization and memory, host memory, and callback durations over the same interval.

Update `scripts/bench/measure_ppo_throughput.py` so it consumes the exact effective campaign configuration rather than hardcoding its own rollout and batch sizes. The benchmark output should identify every effective PPO and environment parameter and provide separate rollout and update timings. Do not compare results unless source, data, config, hardware, warmup, and measurement windows match.

### 2. Find the concurrency knee

On fixed hardware and with a fixed exact worker configuration, compare 1, 3, 6, and 9 workers per GH200, subject to the allocation policy. Use the same data bundle, source, seed policy, and measurement duration. For each point record:

- median and range of per-worker FPS;
- aggregate environment steps/s;
- projected wall time for the frozen campaign budget;
- CPU use and pressure per node;
- GPU utilization and memory per GPU;
- cgroup RAM, process count, and OOM counters.

Choose the worker count that maximizes reliable aggregate steps/s while meeting memory headroom and stability limits. Do not choose based only on the fastest individual worker. Do not assume all 18 workers should share two GPUs if aggregate throughput or training reliability worsens.

### 3. Measure environment parallelism separately

At the selected worker concurrency, compare `n_envs` 16, 32, and 64 using the same bundle and exact environment semantics. Record observation/IPC and rollout timing, process count, aggregate throughput, and memory. Keep total rollout transitions and PPO update semantics explicit: changing `n_envs` while holding configured `n_steps` fixed changes the total samples per update unless the per-environment step count is adjusted. A speed comparison must state whether it holds total rollout size fixed or changes it.

Use the smallest `n_envs` that gives the best measured aggregate campaign throughput without unstable resource use or a material learning-quality regression. Do not infer that a larger environment count is faster simply because it parallelizes more environments.

### 4. Measure reporting overhead without changing training semantics

Time dashboard rendering, periodic deterministic/interim rollouts, checkpointing, and report export separately. The active config requests 20,000-bar dashboard evaluations every 100,000 timesteps, which can accumulate substantial work across a 20M-step run. Measure the actual cost before tuning cadence.

The current interim dashboard path calls policy prediction with `deterministic=False`; evaluation can consume random-number state and affect subsequent training actions. Before changing or isolating that evaluation, preserve and restore all relevant Python, NumPy, PyTorch CPU, and CUDA RNG states, and verify the behavior. The callback's `log_every` also controls active-prior updates, so do not reduce it merely to reduce console output. Separate cheap scalar logging, algorithmically meaningful prior updates, and expensive dashboard evaluation into independent settings with explicit tests and documented defaults.

### 5. Tune PPO compute only as a new experiment

After phase timings identify optimizer work as a significant cost, compare candidate `n_epochs` and `batch_size` values in a controlled pilot. Fewer epochs can reduce update time but changes the number of gradient updates and can change policy quality. Larger batches alter gradient noise and optimizer behavior. Treat each candidate as a distinct configuration and preserve full reporting and acceptance gates.

Do not change rollout size, epochs, batch size, discounting, reward, action semantics, or episode termination in the active campaign. A throughput gain is acceptable only if the candidate completes the required training exposure and passes the agreed training and test diagnostics; speed alone is not success.

## Decision gates

Promote an optimization only when all are true:

1. the benchmark matches the declared source, data, effective config, and hardware;
2. phase timing identifies the work reduced by the change;
3. aggregate steps/s or total wall time improves reproducibly, not just per-worker FPS;
4. no resource instability, data leakage, cache mismatch, or evaluation/training RNG change is introduced;
5. the candidate still completes the intended training period and train/test evaluations;
6. trading behavior and all predeclared survival and reporting gates remain visible and are reviewed.

## Immediate next action

Wait for the live campaign and continuation watcher to reach a terminal state. Then repair the benchmark's hardcoded PPO defaults, run the exact-config phase profile on an authorized GPU allocation, and use those measurements to select the concurrency, `n_envs`, callback, or optimizer experiments. The current evidence supports investigating worker/GPU contention and callback cost; it does not yet justify a single-cause diagnosis or a source-level speed change.
