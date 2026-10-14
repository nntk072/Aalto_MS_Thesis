# 18-Variant Runtime Scaling Screen — 2026-10-14

## Decision

The existing timing evidence does **not** support completing 18 × 20M steps in
one allocation. The supervisor can continue unfinished workers across
six-hour allocations, so the projection below estimates the number of chunks
that might be needed; it does not imply the whole campaign must fit in one
allocation. This is a screening calculation from an existing 100k
shared-bundle resource probe, not a measured runtime for the frozen 20M
campaign.

The only empirical 18-worker timing sample used `n_envs=64`, the shared-bundle
backend, six CPUs per worker, two GH200s on `gpuarm2`, and a 900 GiB cgroup
memory limit. The frozen campaign uses the frames backend, `n_envs=4`, three
CPUs per worker, and requests two GH200s with a six-hour allocation on
`gpuarm1`. The changed backend, environment count, CPU assignment, and node
mean the probe cannot serve as a direct forecast for that campaign.

## Measured training time

All 18 probe attempts completed **106,496 environment steps** for a requested
100,000. PPO rollout batches contain 8,192 steps, so completion rounds up to
the next batch. Across the 18 saved `training_log.json` files:

| Quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| PPO training time for 106,496 steps | 353 s | 567 s | 821 s |
| Effective PPO throughput | 130 steps/s | 188 steps/s | 302 steps/s |
| Linear 20M PPO training time at the same throughput | 18.4 h | 29.6 h | 42.8 h |
| Final deterministic train and test evaluations | 32.6 min | 46.8 min | 49.1 min |
| Linear training time plus one final evaluation pair | 19.0 h | 30.4 h | 43.6 h |

Calculation for each worker:

```text
effective_steps_per_second = completed_steps / ppo_learn_seconds
projected_20M_train_seconds = ppo_learn_seconds * 20,000,000 / completed_steps
```

The final-evaluation pair is measured once after training; it is added once,
not scaled by 200. These estimates omit feature loading, checkpoint/report
export, queue delay, and retries. The wall time of a simultaneous campaign
tracks its slowest worker, rather than the median worker alone.

## Six-hour throughput requirement

The allocation requests six hours, but the controller process is wrapped by a
20,700-second timeout, leaving about **5.75 hours** for the campaign step. A
variant would need at least **966 steps/s** to complete 20,000,000 PPO steps in
that single controller window, before final evaluation/export. That is 5.1×
the measured median and 3.2× the fastest measured worker. At the measured
throughput, linear scaling suggests roughly **5.3 median** or **7.6 slowest
worker** controller windows, before scheduler queue delay; these are only
same-configuration arithmetic screens.

The probe used 64 environments per worker and averaged 66.4% utilization on
both GH200s, with sampled peaks of 97% and 96%. It reached a 251.2 GiB
process-tree PSS peak. The resource sampler did not capture allocation-wide
cgroup memory or process-tree CPU time; its cgroup fields described the
separate sampler step. These measurements do not establish the resource or
throughput behavior of the proposed frames / four-environment configuration.

## Consequence and next gate

Treat total runtime for the `n_envs=4` frames profile as **unvalidated**. The
supervisor's multiple-allocation design can continue the work, but completion
time and queue delay are not established by the `n_envs=64` bundle probe.
Before relying on a duration estimate, a future authorized resource probe
would need to use the exact frozen campaign worker settings and measure PPO
steps/s, final evaluation/export time, allocation-scoped host memory,
process-tree CPU time, and per-GPU utilization. This timing gate does not
supersede the scientific gate: all 18 existing 100k train-equity gates failed,
so the 20M campaign must remain unlaunched.

## Evidence

- 18-worker probe manifest and configuration:
  [`campaign.json`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/campaign.json)
- Checksummed results and resource summary:
  [`campaign_results_summary.md`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/campaign_results_summary.md)
- Raw sampled resources:
  [`resources_gpuarm2.csv`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/logs/resources_gpuarm2.csv)
- Reproducible calculation output:
  [`runtime_scaling_screen_20261014.md`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/runtime_scaling_screen_20261014.md)
  · [`JSON with source hashes and per-worker results`](../../outputs/gpu_campaigns/ladder18_shared_bundle_n64_seed42_100k_resource_probe_20261014/reports/runtime_scaling_screen_20261014.json)
- Calculation source and regression tests:
  [`analyze_campaign_runtime_scaling.py`](../../scripts/matrix/analyze_campaign_runtime_scaling.py)
  · [`test_campaign_runtime_scaling.py`](../../tests/test_campaign_runtime_scaling.py)
- Frozen 20M specification:
  [`campaign YAML`](../../config/campaigns/18_variant_seed42_20m_exploratory_selection_acknowledged_gpu_balanced_20261014.yaml)
- Proposed six-hour allocation:
  [`run_campaign_18way_supervisor.sh`](../../scripts/matrix/run_campaign_18way_supervisor.sh)
