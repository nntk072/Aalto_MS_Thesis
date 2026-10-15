# Fast training branch: GH200 benchmark

> **Plan routing:** Current decisions and next actions are consolidated in the [master project execution plan](../thesis/master_project_execution_plan.md). This file records benchmark evidence and caveats, not a separate active plan.

Date: 2026-10-15  
Branch: `codex/fast-training-wp0`  
Benchmark code: `b847b5b` (`Measure trainer CPU usage in probes`)  
Allocation: Slurm job `20973855`, `gpuarm1`, one GH200, 48 CPUs, 512 GiB RAM

## Scope

These are isolated PPO throughput probes for the existing A3 multi-TP policy. They
exercise the canonical variant resolver, shared bundle, agent builder, and PPO
settings. Each probe trains 262,144 environment steps with seed 42, 64 logical
environments, 128 steps per environment, aggregate rollout 8,192, batch 4,096,
and 15 PPO epochs. The measured device was an NVIDIA GH200 144G HBM3e.

Production evaluation callbacks and checkpoint writes are not included. The
reports therefore compare the PPO core and vector backends; they are not full
campaign completion-time measurements and contain no policy-quality evidence.
No checkpoint was saved and no 18-variant training campaign was launched.

The A3 bundle manifest SHA256 is
`3ddea0b613e7096314263010c24c2665eed8d782d5db5c74d9742dbd3cb0f69a`.
The measured backend is the only configured difference in each pair.

## A3 results

Each value is the mean of two runs. The run-level JSON files are under
`outputs/benchmarks/fast-training-wp0/`.

| Backend | Logical envs | Physical workers | Steps/s | Ready cgroup RAM | Trainer-tree PSS |
|---|---:|---:|---:|---:|---:|
| SubprocVecEnv | 64 | 64 | 2,249.5 | 10.90 GB | 9.32 GiB |
| ShardedBundleVecEnv, 16 shards | 64 | 16 | 2,193.2 | 6.40 GB | 4.98 GiB |

Sharding reduced ready-state cgroup RAM by about 41% and process-tree PSS by
about 47%. Its mean throughput was about 2.5% lower in this A3 comparison. The
two-run spread and short duration do not support treating that small difference
as a stable speed regression or gain.

One post-change run measured process-tree CPU use at 2.23 average cores for
SubprocVecEnv and 1.82 for sharded-16, with 48 CPUs allocated. These are single
run observations, not a sustained node-wide utilization profile. The other
reports predate CPU accounting and have no CPU measurement.

## Provisional trainer-concurrency sweep

Using the same A3 profile and 16-shard backend, I ran 1, 2, 4, 8, and 9
checkpoint-free probes together on the same GH200. All trainers used 64 logical
environments and 262,144 steps. The multi-trainer points use separate seeds.
Aggregate rate is total completed environment steps divided by the common
start-to-finish wall window.

| Concurrent trainers | Aggregate steps/s | Mean steps/s per trainer | Ready cgroup RAM | Sum CPU cores | Sum trainer-tree PSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,193 | 2,193 | 6.40 GB | 1.82 | 4.98 GiB |
| 2 | 3,701 | 1,970 | 9.05 GB | 3.54 | 9.70 GiB |
| 4 | 6,210 | 1,647 | 14.43 GB | 6.98 | 19.04 GiB |
| 8 | 7,885 | 1,032 | 25.96 GB | 12.23 | 37.62 GiB |
| 9 | 7,983 | 929 | 27.98 GB | 13.34 | 42.46 GiB |

The 8- and 9-trainer `nvidia-smi dmon` samples averaged 97.9% and 98.4% SM
utilization while the GPU was active. The ninth trainer increased measured
aggregate throughput by only about 1.3% over eight, while reducing per-trainer
rate by about 10%. CPU use remained below 14 of 48 cores, and the nine-trainer
ready cgroup memory sample was about 28 GB of the 512 GiB allocation. No probe
failed or exhausted GPU memory.

At nine concurrent trainers, the paired backend comparison was:

| Backend | Aggregate steps/s | Per-trainer steps/s | Ready cgroup RAM | Sum trainer-tree PSS | Sum CPU cores |
|---|---:|---:|---:|---:|---:|
| SubprocVecEnv | 7,748 | 925 | 70.78 GB | 81.74 GiB | 16.24 |
| ShardedBundleVecEnv, 16 shards | 7,983 | 929 | 27.98 GB | 42.46 GiB | 13.34 |

In this single paired run, sharding had about 3% higher aggregate throughput,
about 61% lower ready cgroup RAM, and about 48% lower summed PSS. Both backend
samples had the GPU near saturation. Treat the small throughput edge as
provisional until repeated; the memory and worker-count reductions are the
clearer result.

This makes nine sharded trainers per GH200 a plausible makespan choice for the
18-variant campaign (nine per node on two GH200 nodes): it lets all 18 variants
progress together, while the ninth slot adds little aggregate GPU throughput.
That is still a provisional projection from one GH200 and a 262k-step core
benchmark. It does not include production callbacks, full 20M schedules, two-node
interference, or confirm a 20M allocation will fit its wall limit. Re-measure
with the selected learning `n_envs` before the actual campaign.

At the measured nine-trainer per-worker rate of about 929 steps/s, a 20M-step
variant projects to roughly 6.0 hours of PPO core time. That leaves no practical
margin in a six-hour allocation for bundle checks, evaluation, callbacks, and
checkpoint promotion. Plan for a 12-hour allocation or a validated allocation
resume path; do not treat this short probe as evidence that a 20M campaign fits
in six hours.

Earlier A0-core probes on the same GH200 measured 3,026 steps/s for SubprocVecEnv
and 3,132 steps/s for sharded-16, while ready-state RAM fell from 10.56 GB to
6.04 GB. This different policy profile shows that backend throughput effects
depend on the PPO workload; it does not establish an A3 speedup.

## Decision

- Keep `subproc` as the default so existing training behavior stays unchanged.
- Keep sharded bundle execution opt-in. It is a demonstrated memory-saving
  option, but A3 did not show a material single-trainer throughput improvement.
- Do not add compact observation transport yet. The A3 PPO update phase used
  about 60% of wall time, and these measurements do not show that observation
  IPC is the remaining dominant bottleneck. The plan's 10% sustained-throughput
  threshold for that added complexity has not been met.
- Use nine concurrent trainers per GH200 as the current measured candidate if
  the campaign must run all 18 variants together. Revalidate at the `n_envs`
  chosen by the learning-quality track, because changing it changes rollout
  geometry and resource use.

## Verification and artifacts

At commit `b847b5b`, full-tree Ruff formatting and lint passed, staged full-tree
Mypy passed with no issues in 417 source files, and
`tests/test_bench/test_measure_ppo_throughput.py` passed (5 tests). The full test
suite had passed earlier on this branch (1,502 tests). Raw JSON and logs are
local ignored benchmark artifacts, not versioned source data.
