# Triton GPU Training Agent Plan

Repository: /scratch/work/nguyenl37/Aalto_MS_Thesis
Status: Controller implementation and mock integration verified on Triton GH200, 2026-10-10. Production campaign remains gated on end-to-end benchmark and complete CI.

## Training matrix

Use all 18 entries in scripts/variant_list.txt (6 baseline, 6 PO3, 6 distribution). Train seeds 42, 43, 44, 45, 46 for each: **90 unique runs**, each **20,000,000 timesteps** (1.8 billion total). The 11 additional variants in config/experiments.yaml are outside the core campaign until explicitly selected. Use quant_rl/train/variant_resolver.py to resolve each configuration. Keep feature, reward, action, trading cost, SL/TP, split and OOS evaluation semantics intact.

## Source code audit and proposed edits

Inspect scripts/matrix/run_18variant_matrix.sh, scripts/matrix/run_18variant_matrix_gh200_tmux.sh, scripts/train/train_one_variant.sh, config/experiments.yaml, launch/recovery documentation and AGENTS.md before editing. Resolve fixed SEED=42, inconsistent 100k/20M defaults, and hard-coded trainer concurrency. Refactor scripts into configurable bounded worker queue with --variant-file, --seeds, --timesteps, --workers, --n-envs, --campaign-id, --dry-run, --resume, and --benchmark switches. Do not invoke srun, salloc, sbatch, scancel, or start/stop allocations implicitly. Respect Triton Slurm rules; never run GPU training on login nodes.

Dry-run must enumerate exactly 90 distinct keys (<variant>__seed<seed>), compute total timesteps, resolve configs, verify inputs, and launch no jobs. Use a per-campaign lock, atomic state updates and append-only queue.jsonl events. Persist run status (pending/running/succeeded/failed/interrupted), immutable attempt count, PID, host, Slurm job ID, GPU, git SHA, config digest and checkpoints. Validate checkpoint correctness before resume; do not double-count finished runs or overwrite successful run artifacts. Gracefully handle SIGTERM, preemption, disk-full, OOM, Slurm timeouts, retry backoff and worker crash.

## Resource discovery on Triton

Before deciding concurrency, inspect squeue/sinfo/scontrol, actual allocation, scratch quota, GPU usage, CPU cpuset, cgroup memory.max/current/peak/events, process PSS, disk throughput and existing trainers. Use allocation caps, not free -h node totals.

Reference node sizes only (not guaranteed allocations): GH200 2 GPUs / ~1.1 TB RAM / 144 CPUs (aarch64 .venv); H200 8 GPUs / ~1.98 TB RAM / 128 CPUs (x86_64 .venv-x86); H100 4 GPUs / ~990 GB RAM / 192 CPUs (x86_64 .venv-x86). Verify Python, CUDA and architecture on compute node.

Existing *environment-only* benchmark:
| Vector envs | PSS | Env steps/s |
| ---: | ---: | ---: |
| 8 | 18.5 GiB | 17,483 |
| 16 | 34.3 GiB | 19,640 |
| 32 | 65.9 GiB | 24,598 |
| 64 | 129.2 GiB | 26,639 |

Bundle environment at 64 envs: ~7.6 GiB, 27,751 env steps/s, but parity is unverified. Do not use in scientific production absent proof that transitions, observations, rewards and termination semantics match.

Benchmark *end-to-end PPO training* (rollout + gradient optimization + writing checkpoints/metrics), not just environment stepping. Try worker counts 1/2/3/4/6 at 8 envs, and 2/3/4 at 16 envs when allocation allows. Illustrative allocation profiles, all subject to measurement: (workers, CPUs, RAM GiB) = (1,16,96), (2,32,192), (3,48,256), (4,64,384), (6,96,512) at 8 envs; (2,48,256), (3,72,384), (4,96,512) at 16 envs. Maintain at least 64 GiB node reserve and >=35% headroom against observed RAM peak, respecting stricter cgroup limits. Leave CPU capacity for controller, logging and Slurm; monitor GPU VRAM pressure. Pick worker count that maximizes sustained *aggregate PPO trained timesteps/hour* without CPU/GPU/IO oversubscription. Current 3 x 64-env trainer approach may overload 32 allocated CPUs.

## Durable tmux/session/worker logs (mandatory)

Persist ALL logs on Triton scratch, **never solely in tmux scrollback**, under:

```text
outputs/gpu_campaigns/<campaign_id>/
  campaign.json
  queue.jsonl
  logs/
    controller.log
    allocation.log
    scheduler.jsonl
    tmux.log
    resources.csv
    benchmarks/<profile>.log
    workers/<variant>__seed<seed>__attempt<N>.log
    failures/<variant>__seed<seed>__attempt<N>.log
    failures/summary.jsonl
  runs/<variant>__seed<seed>/
    resolved_config.json
    provenance.json
    checkpoints/
    metrics/
  markers/<variant>__seed<seed>.done
  reports/campaign_status.json
  reports/benchmarks.csv
```

Controller must append stdout+stderr to controller.log, e.g. Bash with pipefail:
```bash
set -o pipefail
run_controller 2>&1 | tee -a "$LOG_DIR/controller.log"
rc=${PIPESTATUS[0]}
exit "$rc"
```

Create named tmux session and append session/pane lifecycle and capture/pipe output to tmux.log (e.g., tmux pipe-pane -o). Every trainer's stdout+stderr must append to a *unique attempt-specific* log. Never truncate previous attempts. Log UTC start/end, command, variant, seed, attempt, host, job ID, PID, GPU ID, git SHA, resolved config hash, elapsed time and process exit code. Capture failure traces in failure log plus failures/summary.jsonl. Keep append-only queue transitions and atomic status snapshots. Record resource samples every ~30-60 seconds: timestamp, CPU, process PSS, cgroup memory.current/peak/events, GPU utilization and VRAM, scratch free/quota and I/O. Make logs readable after SSH disconnect, tmux detach, process crash, job walltime or controller restart.

Diagnostics, from Triton:
```bash
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
tmux ls
tail -n 150 outputs/gpu_campaigns/<campaign_id>/logs/controller.log
tail -n 150 outputs/gpu_campaigns/<campaign_id>/logs/workers/<variant>__seed<seed>__attempt1.log
tail -n 100 outputs/gpu_campaigns/<campaign_id>/logs/tmux.log
cat outputs/gpu_campaigns/<campaign_id>/reports/campaign_status.json
```

Test persistence with tmux detach and SSH disconnection, a fake crashing trainer, controller restart and resume. Verify original failure traceback and exit code are still on disk. Keep worker logs and artifacts for failed runs.

## Execution phases and gates

1. **Inspect (read-only):** git status and guidance, current Slurm state, launch scripts, variant config resolver, checkpoint/resume path and actual resources.
2. **Implement + dry-run (no GPU):** bounded queue, disk logging, resilient resume, state management, monitors and CLI. Dry-run 90 unique runs with no Slurm side effects.
3. **Linux CI:** uv run ruff format --check . ; uv run ruff check . ; uv run mypy . ; targeted pytest for resolver, manifests, retries and resume; bash -n changed scripts; fake-worker integration tests. Fix code quality while preserving research semantics.
4. **Benchmark (requires approved compute allocation):** run measured PPO throughput profiles, capture complete metrics and logs, select worker count from actual allocation.
5. **Pilot (requires authorization):** representative baseline, PO3 and distribution runs; verify checkpoint/resume, logs, training integrity and split isolation.
6. **Production (separate explicit authorization):** run 18 variants x five seeds x 20M under named tmux campaign with continuous on-disk monitoring; recover failed runs safely.
7. **Audit:** verify 90 successful run keys and provenance, validate checkpoint/metrics, generate comparisons and OOS results from real artifacts. Preserve historical failed-attempt logs.

## Agent acceptance criteria

- Exactly 90 correctly resolved distinct production keys and 1.8B timesteps in an auditable dry-run.
- Benchmark-backed dynamic concurrency selected within real Slurm cgroup CPU/RAM/GPU limits.
- tmux, controller, worker, scheduler, Slurm allocation, failure and resource logs persist on Triton disk.
- Logs capture stderr and original process exit codes and survive disconnect, restart and failed attempts.
- Restart never silently overwrites completed artifacts or double-schedules the same run.
- Linux format/lint/type, shell syntax and relevant integration gates pass before production.
- Production launch and Slurm allocations await explicit authorization.


## Compute-node-only agent workspace and architecture (mandatory)

All implementation, formatting, mypy, pytest, benchmarks, and training for this campaign must execute from a **tmux session on an allocated GPU compute node**, including tasks that do not need CUDA. Use login solely for SSH, read-only Slurm/tmux discovery, inspecting existing logs, and the initial remote document update. Never run project tooling in Windows or substitute a login-node Python environment.

At the beginning of every compute session, run these checks inside the tmux pane:

```bash
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
hostname
uname -m
printf 'SLURM_JOB_ID=%s\\n' "${SLURM_JOB_ID:-unset}"
source scripts/triton/activate_venv.sh
python -c 'import platform,sys; print(platform.machine(),sys.executable)'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

- GH200 is `aarch64`; use its native `.venv`. H100/H200 are `x86_64`; use `.venv-x86`. `scripts/triton/activate_venv.sh` is the canonical selector. Confirm the effective interpreter **inside the compute pane**; do not rely on login-node `uname -m`.
- Before using tmux, run `bash scripts/triton/status.sh` and `bash scripts/triton/find_gpu_session.sh` from the login node to locate an existing compute allocation and pane. A stale tmux session on a login node is **not** a compute allocation. Inspect pane host and Slurm job before reusing it.
- Repository rules in `AGENTS.md` and `.agents/rules/triton-slurm.md` require separate explicit user permission for **any new** `srun`/Slurm step or `scancel`. Preparing this file authorizes neither. If no allocation is running, stop before allocation and request specific permission with proposed GPU model/count, CPU, memory and walltime.
- Once an approved allocation is available, attach to its existing tmux compute pane or start one tmux **inside the allocated compute shell**, named `gpu-campaign-<campaign_id>`. Avoid making a tmux window per run. The bounded controller dispatches workers within allocated CPU, RAM and GPU limits. Do not launch extra Slurm steps with `srun --overlap`.
- Create `outputs/gpu_campaigns/<campaign_id>/logs/` on scratch **before** launching the tmux-controlled command. Start `tmux pipe-pane -o -t <session>:<pane> 'cat >> <absolute-scratch-path>/logs/tmux.log'` or an equivalent append-only capture, then run the controller with `tee -a` as described above. The controller, individual attempts and resource poller each have their own durable on-disk logs. A tmux session alone is not sufficient logging.
- On detachment and reconnection, inspect `tmux ls`, `tmux capture-pane`, `logs/tmux.log`, `logs/controller.log`, and attempt logs to verify the processes and audit trail. Preserve campaign files across allocation expiry and reruns. Log the actual compute hostname, architecture, venv path, Slurm allocation, CPU set and cgroup limits in campaign.json.

**Updated preflight (2026-10-10):** Slurm job `20877341` allocated one NVIDIA GH200 on `gpuarm1.int.triton.aalto.fi` (aarch64), 32 CPUs, 384 GiB, six hours. Source implementation is `scripts/matrix/campaign_controller.py`. Dry-run enumerates 90 keys, tests confirm durable attempts and restart behavior, and 12 resolver tests pass. Ruff tree gates pass; full-tree mypy and end-to-end training benchmark must be verified before production.
