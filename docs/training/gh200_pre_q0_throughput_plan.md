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
   throughput before making this a production default. The implementation is
   in `quant_rl/models/rollout_buffer.py`; after two consistent paired GH200
   measurements and exact minibatch parity, it is enabled by default. Set
   `ppo.rollout_buffer_on_device=false` to retain the CPU-buffer control.

3. **Try pinned host memory and asynchronous prefetch. — Closed without a new path**
   Full rollout staging already removes repeated minibatch transfers. The
   earlier pageable-transfer profile measured about 134 ms of summed GPU copy
   time across three PPO updates, far below the update and rollout phases.
   Pinned prefetch would add a second data path to optimize a small residual;
   it was not implemented because there is no evidence it can materially
   improve this profile.

4. **Sweep PPO batch and rollout geometry. — Measured; reference retained**
   Compare larger minibatches and batched rollout sizes (`n_envs × n_steps`),
   including optimizer-step counts and policy-quality effects. These settings
   change PPO update geometry, so retain the current configuration as the
   reference until training/evaluation quality is reviewed.

5. **Reduce host dispatch, synchronization, and avoidable tensor-layout work. — Measured**
   The first trace recorded 93,639 CUDA launch calls, 11,742 `aten::item` calls,
   and about 490 ms summed cuDNN NCHW/NHWC conversion-kernel time in the
   three-update window. Map these events to source and test whether they cause
   meaningful wall time before attempting fusion or layout changes.

6. **Try `torch.compile` on the learner. — Blocked by node toolchain**
   Profile compilation and steady-state separately; test graph breaks and
   numerical behavior before comparing throughput.

7. **Try CUDA Graph capture / compile reduced-overhead mode. — Not run**
   Do this after shapes and memory addresses are stable, and compare against the
   eager/compiled results with identical update geometry.

8. **Tune TF32 and BF16 use. — Measured; current precision retained**
   TF32 and BF16 encoder autocast are already enabled in this code path. Treat
   the current settings as the baseline; test any broader mixed precision for
   both speed and numerical stability. FP8 is not an assumed improvement.

9. **Consider a GPU-vectorized TradingEnv. — Not justified by measured phase time**
   Proceed only if phase timing shows environment simulation dominates. This is
   a larger rewrite and requires exact transition, fill, risk, and reward parity.

10. **Evaluate a JAX/XLA implementation as a separate high-risk option. — Deferred**
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
prototype. New PPO runs use the device buffer by default, including the pending
18-variant job when it starts. The throughput comparison is still PPO-core
only; an end-to-end campaign and training/evaluation quality review remain open.

## Follow-up experiments on 2026-10-15

All runs below used the existing single GH200 allocation `20973855` on
`gpuarm1`; no new `srun` or Slurm step was started. The corrected geometry runs
forced `ShardedBundleVecEnv` with 16 shards, 64 logical environments, seed 42,
and 262,144 environment steps. Their A3 reference was 128 steps per environment,
batch size 4096, and 15 epochs. The first batch-sweep attempt accidentally used
`SubprocVecEnv`; those files are retained as diagnostic artifacts but are not
used as reference-backend results.

| Rollout steps/env | Batch | Mean steps/s | Update seconds | Peak allocated VRAM | Optimizer steps/run |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2048 | 3,067 | 36.17 | 2.21 GiB | 1,920 |
| 128 | 4096 | 3,519 (two runs) | 26.00 | 3.62 GiB | 960 |
| 128 | 8192 | 3,697 (two runs) | 22.82 | 6.43 GiB | 480 |
| 256 | 4096 | 3,428 | 26.26 | 4.32 GiB | 960 |
| 256 | 8192 | 3,813 | 22.70 | 7.13 GiB | 480 |
| 512 | 16384 | 3,805 | 22.04 | 14.15 GiB | 240 |
| 512 | 32768 | 3,766 | 22.03 | 25.41 GiB | 120 |

The 8192 batch improved reference-geometry throughput by about 5.1% across
these two runs, while cutting the number of optimizer steps in half. The
512/16384 and 512/32768 configurations used more VRAM but made only 240 and 120
optimizer steps, compared with 960 at the reference geometry. These are
throughput probes, not policy-quality results. No larger batch or rollout
geometry is adopted until training and evaluation metrics are compared under a
quality-preserving protocol.

The benchmark now counts actual optimizer `step()` calls. Some earlier
corrected-backend JSON files predate that field fix: their `optimizer_steps`
value is SB3's completed PPO epoch counter, not optimizer calls. The table uses
the effective rollout/minibatch geometry to state the optimizer-step count.

| Workstream | Evidence and decision |
| --- | --- |
| Pinned host memory / asynchronous prefetch | The original profile measured about 134 ms of summed GPU copy time across three PPO updates. Full-rollout VRAM staging already removes repeated minibatch copies, so pinned prefetch was not added. |
| Host synchronization reduction | A temporary PPO implementation deferred metric scalar reads and passed exact CPU parameter parity in the focused CartPole test. Two GH200 runs measured 3,537 and 3,568 steps/s; the gains were within the reference run spread. The implementation and config switch were removed rather than adding an unproven production code path. Artifacts: `gh200-a3-deferred-metric-sync-r1.json` and `r2.json`. |
| `torch.compile` / CUDA Graphs | The GH200 probe failed before training: `Failed to find C compiler` while Inductor initialized Triton's CUDA helper. No `cc`, `gcc`, or `clang` executable or compiler module was available in the compute shell. PyTorch documents `reduce-overhead` as the mode that uses CUDA Graphs for eligible graphs; that path could not be evaluated without the compiler toolchain. |
| TF32/BF16 | The production path already enables TF32 and BF16 encoder autocast. A `medium` float32 matmul run measured 3,576 steps/s versus 3,581 at `high`, with no gain. The higher-risk full-policy BF16 and FP8 variants were not enabled. |
| GPU-vectorized environment | Direct per-environment timing on the 16-shard backend measured a mean `TradingEnv.step()` duration of about 0.34 ms. The maximum summed step time on any shard was 6.09 seconds for the 262,144-step run, versus 50.20 seconds for the full rollout/framework phase. A GPU environment rewrite is not justified by this share. Artifact: `gh200-a3-sharded-geometry-r256-b4096.json`. |
| JAX/XLA | No JAX implementation or dependency is present in the project. A faithful port would include the trading environment and execution semantics, not just the PPO learner. It remains a separate high-risk project rather than a throughput change in this branch. |

The only production fast-path change in this branch remains full-rollout device
staging, enabled by default with a CPU-buffer control. PPO minibatch parity is
tested exactly. No full production training/evaluation campaign was run during
these probes; no Q0 or 18-variant campaign was started by this optimization
work. Therefore these measurements do not establish that a policy survives a
full-year training episode or that its evaluation performance is preserved.
