# GH200 Training Throughput Plan (Pre-Q0)

## Goal and boundaries

Improve measured PPO training throughput on the GH200 without changing the
trading objective, execution rules, or evaluation protocol. Keep the current
A3 profile as the reference. Do not start Q0 or the 18-variant campaign as part
of this optimization work. Use the existing allocation for isolated probes;
do not request another Slurm allocation for this plan.

Every speed change must be measured against the same A3 variant, 64 logical
environments, 16 shared-bundle shards, 128 rollout steps per environment,
batch size 4096, and 15 PPO epochs. Compare unprofiled throughput across repeat
runs; profiler-instrumented FPS is diagnostic only. A candidate is accepted
only if PPO sampling/update semantics remain understood and training/evaluation
quality does not regress.

## Ordered work

1. **Profile one trainer end to end. — Complete**
   Capture PPO CPU/CUDA operators, host-to-device transfers, GPU activity,
   memory use, environment rollout time, and PPO update time. Keep campaign
   callbacks out of the microbenchmark and identify their timing separately.

2. **Stage one complete PPO rollout in VRAM. — Prototype measured**
   Upload the completed rollout once, then reuse it across PPO epochs. Preserve
   the same per-epoch permutation and minibatch contents. Measure memory use and
   throughput before making this a production default. The opt-in prototype is
   implemented in `quant_rl/models/rollout_buffer.py`; the default remains the
   existing CPU buffer pending campaign/evaluation review.

3. **Try pinned host memory and asynchronous prefetch.**
   Use this as a fallback if full GPU staging is awkward or if overlap improves
   throughput. Confirm transfers actually overlap learner work.

4. **Sweep PPO batch and rollout geometry.**
   Compare larger minibatches and batched rollout sizes (`n_envs × n_steps`),
   including update count and policy-quality effects. These settings change PPO
   update geometry, so retain the current configuration as the reference.

5. **Reduce host dispatch, synchronization, and avoidable tensor-layout work.**
   The first trace recorded 93,639 CUDA launch calls, 11,742 `aten::item` calls,
   and about 490 ms summed cuDNN NCHW/NHWC conversion-kernel time in the
   three-update window. Map these events to source and test whether they cause
   meaningful wall time before attempting fusion or layout changes.

6. **Try `torch.compile` on the learner.**
   Profile compilation and steady-state separately; test graph breaks and
   numerical behavior before comparing throughput.

7. **Try CUDA Graph capture / compile reduced-overhead mode.**
   Do this after shapes and memory addresses are stable, and compare against the
   eager/compiled results with identical update geometry.

8. **Tune TF32 and BF16 use.**
   TF32 and BF16 encoder autocast are already enabled in this code path. Treat
   the current settings as the baseline; test any broader mixed precision for
   both speed and numerical stability. FP8 is not an assumed improvement.

9. **Consider a GPU-vectorized TradingEnv.**
   Proceed only if phase timing shows environment simulation dominates. This is
   a larger rewrite and requires exact transition, fill, risk, and reward parity.

10. **Evaluate a JAX/XLA implementation as a separate high-risk option.**
    Prototype only after the lower-risk experiments. Compare the same trading
    semantics, PPO geometry, and evaluation protocol; include porting and compile
    costs in the comparison.

Population-based training is an algorithm-selection/tuning method, not a direct
per-trainer throughput optimization. Keep it as a separate research experiment
if the goal later expands from faster fixed-configuration training to faster
hyperparameter search.

## First measurement

The existing unprofiled A3 shared-bundle run at the reference geometry processed
262,144 environment steps in 120.32 seconds (2,178.8 steps/s). PPO updates took
69.58 seconds (57.8% of that run); rollout and framework overhead took 50.73
seconds. Two new paired, unprofiled runs processed 262,144 steps per run:

| Buffer | Run 1 | Run 2 | Mean steps/s | Mean PPO update time | Peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing CPU `DictRolloutBuffer` | 2,135.9 steps/s | 2,144.2 steps/s | 2,140.1 | 70.55 s | 2.91 GiB |
| Opt-in `DeviceDictRolloutBuffer` | 3,374.8 steps/s | 3,433.8 steps/s | 3,404.3 | 26.27 s | 3.62 GiB |

The measured mean throughput gain is 59.1%; mean PPO update time fell 62.8%.
Rollout/framework time stayed similar (51.94 s CPU versus 50.74 s device
buffer), which is consistent with the optimization targeting minibatch data
staging. Peak allocated device memory rose by about 0.71 GiB, remaining well
below the GH200's 144 GiB. These are two PPO-core pairs on one allocation, not
an end-to-end campaign result or a multi-node concurrency result.

The buffer uses the same NumPy permutation and produced exactly matching
minibatch tensors in the focused parity test
(`tests/test_models/test_rollout_copy.py`). This supports sample-order and data
parity; it is not a claim that training quality or evaluation results improved.

A PyTorch CPU/CUDA trace was then captured on the same GH200 and bundle for 64
environments, 16 shards, 128 steps per environment, batch size 4096, and 15
epochs. The diagnostic run processed 81,920 steps in 51.32 seconds (1,596.1
steps/s); profiler overhead makes this unsuitable as a throughput comparison.
The trace covered three PPO update calls and recorded:

- 36.1 GB of pageable host-to-device copies over those three updates, in 1,881
  copy events; the GPU copy events sum to about 134 ms.
- 106,635 GPU kernel events, with about 1.78 seconds of summed kernel time.
  Kernel and operator durations overlap, so these totals are not additive wall
  time. One-second `nvidia-smi` samples over the diagnostic run averaged 12.6%
  GPU utilization and peaked at 40%; samples include rollout and profiler
  warm-up periods.
- Peak device memory in the diagnostic run was about 4.3 GiB on a 144 GiB GH200.
- The trace contains many CUDA launch calls and cuDNN NCHW/NHWC conversion
  kernels. These make host dispatch and TCN layout handling candidates to
  investigate; the current trace does not prove either is the dominant wall-time
  bottleneck.

The copy volume makes rollout staging worth benchmarking, but its measured GPU
copy duration is small relative to the unprofiled 120-second run. Do not claim a
large speedup from caching alone. The low sampled GPU utilization and layout/
launch activity make phase-level follow-up profiling important.

## Evidence files

- Unprofiled reference: `outputs/benchmarks/fast-training-wp0/gh200-a3-sharded-16-r1.json`
- Profiled run metrics: `outputs/benchmarks/fast-training-wp0/gh200-a3-single-profiled.json`
- Compressed Chrome trace: `outputs/benchmarks/fast-training-wp0/gh200-a3-single-chrome-trace.json.gz`
- One-second GPU samples: `outputs/benchmarks/fast-training-wp0/gh200-a3-single-gpu-samples.csv`
- CPU buffer repeat runs: `outputs/benchmarks/fast-training-wp0/gh200-a3-cpu-rollout-buffer-r1.json`, `outputs/benchmarks/fast-training-wp0/gh200-a3-cpu-rollout-buffer-r2.json`
- Device buffer repeat runs: `outputs/benchmarks/fast-training-wp0/gh200-a3-device-rollout-buffer-r1.json`, `outputs/benchmarks/fast-training-wp0/gh200-a3-device-rollout-buffer-r2.json`
Q0, existing campaign processes, and Slurm job states were not changed by this
prototype. The optimized buffer is opt-in; a campaign-level adoption decision
and an end-to-end training/evaluation comparison remain open.
