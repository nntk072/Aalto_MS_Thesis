# Fast training branch: GH200 benchmark

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
- Defer the final trainers-per-GPU and makespan choice until the learning-quality
  track selects its `n_envs`; changing that value also changes rollout geometry.
  A future concurrency benchmark should compare aggregate useful steps/s and
  memory at that frozen setting.

## Verification and artifacts

At commit `b847b5b`, full-tree Ruff formatting and lint passed, staged full-tree
Mypy passed with no issues in 417 source files, and
`tests/test_bench/test_measure_ppo_throughput.py` passed (5 tests). The full test
suite had passed earlier on this branch (1,502 tests). Raw JSON and logs are
local ignored benchmark artifacts, not versioned source data.
