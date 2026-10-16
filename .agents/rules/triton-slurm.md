# Triton / Slurm (GPU) — agent rules

Aalto Triton cluster. **Canonical workdir:** `/scratch/work/nguyenl37/Aalto_MS_Thesis`.

Helper scripts: [`scripts/triton/`](../../scripts/triton/) (`status.sh`, `find_gpu_session.sh`, `attach_or_alloc.sh`).

## WorkDir

- All training, eval, and agent edits for GPU runs must use
  `/scratch/work/nguyenl37/Aalto_MS_Thesis` (real scratch path).
- A symlink `~/Aalto_MS_Thesis` → that scratch path is OK; always `cd` to the
  scratch path (or resolve the symlink) before any GPU-side command.
- Do **not** treat a WSL or laptop checkout as the live Triton tree.

## Hard rule: `srun` and `scancel` only with user confirmation

Agents **may** run `srun` / `attach_or_alloc.sh`, but **only after the user
explicitly permits a new allocation in this chat** (e.g. “ok to srun”,
“allocate a GPU”, “start the job”). Do **not** self-start allocations.

Agents **may** run `scancel`, but **only after the user explicitly permits
cancelling that job in this chat** (e.g. “scancel”, “cancel the GPU job”,
“cancel this allocation”). Do **not** cancel a running or completing job on
your own.

Access is not permission to allocate. A login shell, SSH onto a node that
already has this user's job, a live tmux pane, or “use the current GPU task”
does **not** allow `srun`. These stay blocked until the user asks for a
**new** allocation:

- `srun` (batch, `--pty`, or interactive)
- `srun --overlap`, `srun --jobid=<id>`, or any extra Slurm step
- `attach_or_alloc.sh` when it would allocate

Retrain, delete outputs, restart, or “the run is wrong” is **not** permission
to `scancel`. Those stay blocked until the user names the cancel:

- `scancel <jobid>` or `scancel` of a running train
- `REPLACE=1` on `scripts/triton/two_trains_one_h200.sh` (it cancels other GPU jobs)
- Replacing a job after OOM, a code change, or a bad log

If a job was cancelled without that permission, do **not** `scancel` again.
A replacement `srun` still needs its own explicit allocation permission.

Stopping the foreground program inside an existing task, when the user says
to stop it and continue, is not `scancel` and is not a new `srun`. Continue
on that same node (SSH `step_extern` or `tmux send-keys`). Do not replace
the job.

Triton emails when many jobs last &lt; ~120s. Prefer one long shell (hours),
never a burst of short diagnostic jobs.

### Short-job / step monitor hard gate

The cluster monitoring rule is operationally strict: a job or step with an
elapsed time below **120 seconds** counts as short. Repeated short jobs/steps
can trigger an account warning even when they are launched under one parent
allocation or spread across nodes. Treat the reported situation (20 or more
short jobs/steps, especially across two or more nodes) as a launch-control
failure, not as an acceptable cost of campaign startup. Recent examples were
job `20982346` (46 seconds, 2 steps across 2 nodes) and job `20982372` (110
seconds, 22 steps across 2 nodes); these are incident evidence, not acceptable
templates for future launches.

Before starting or resuming any campaign, benchmark, test batch, or supervisor,
inspect its launch path and establish the maximum number of Slurm jobs/steps it
can create, including retries, health checks, worker fan-out, and shutdown. The
default permitted count of **new short Slurm jobs/steps is zero**. Do not use
one `srun` per worker, variant, retry, test, monitor, or node-local command.
Run short commands directly in the login shell when CPU-only and permitted, or
inside an already-running allocation without creating another Slurm step.
For multi-node work, use only the already allocated node shells/persistent
supervisors; do not fan out short steps across nodes. If the design cannot meet
this rule, do not launch it: first change the launcher to a bounded number of
persistent long-lived steps or direct child processes, then verify the topology
before resuming. A parent job lasting hours does not make its short child steps
compliant.

After launch, inspect `squeue`/`sacct` step records and supervisor logs from the
login node. If unexpected steps appear or any worker repeatedly exits before
120 seconds, stop submitting/retrying work, preserve logs and allocation state,
and diagnose the launcher. Do not cancel the allocation unless the user
explicitly authorizes cancellation. Never attempt to evade the monitor by
keeping a short command alive artificially; use a genuine persistent task or
run the command inside an existing shell.

### Default workflow (no new job)

1. `bash scripts/triton/status.sh` (or `find_gpu_session.sh`) on login.
2. If a GPU tmux/`srun` already exists → reuse it (`tmux send-keys`); do not
   allocate another node.
3. If none exists and CUDA is required → **ask the user** before any
   `srun` / `attach_or_alloc.sh`.

### Allowed without asking

- Login SSH: edit files, `git`, `grep`/`rg`, read `outputs/` logs.
- `squeue -u $USER`, `status.sh`, `find_gpu_session.sh` (inspect only).
- Availability probes on login (see below) — these are **not** jobs.
- `tmux capture-pane` / list sessions; send commands into an **existing**
  GPU pane.

### Requires explicit user confirmation in-chat

- Any new `srun` (interactive, batch, `--overlap`, or `--jobid=`).
- `bash scripts/triton/attach_or_alloc.sh` when it would start a new alloc.
- `scancel` of a running or completing job, including `REPLACE=1`.
  “Retrain” or “delete the old files” is not this permission.
- Extra GPU jobs for “quick” pytest / ruff / mypy / CUDA smoke when no
  reusable shell exists (prefer asking for one long alloc, or login-only
  x86-safe tools).

### Still forbidden

- Any loop/burst that submits short Slurm jobs or steps, including one `srun`
  per variant, retry, test, or monitor. A task expected to take under 120s must
  run as an ordinary command inside an already-running allocation; it must not
  create a new job or step.
- Launching a campaign worker through a separate Slurm step for every variant
  when a persistent supervisor or direct child process can run it inside the
  existing allocation. Pack workers into the fewest long-lived steps required
  for GPU binding; do not turn each queue poll or retry into a step.
- **Known launcher hazard:** `campaign_controller.py --worker-nodes ...` builds
  one `srun --overlap` step per worker. Do not use that mode to launch a large
  multi-worker campaign until its step topology is changed to one persistent
  node supervisor per node (with trainer processes as children) or an
  equivalent packed design. Before any campaign, inspect the actual launch
  command and estimate the number of steps; reject a design that can fan out
  dozens of short-lived steps during startup/failure recovery.
- Rapid automatic retry after a worker exits early. Preserve its log, let the
  supervisor record the failure, inspect the cause, then resume through the
  campaign controller only after confirming that the retry is a sustained run.
- Parallel one-off benchmark jobs on multiple nodes for work that fits the
  current allocation. Prefer one allocation, one sampler, and a fixed benchmark
  window long enough to include warm-up and steady-state throughput.
- `scancel -u $USER` unless the user explicitly requests a full wipe.
- Using `srun` for login-node checks (`ls`, log tails, plain `grep`).
- GPU count limits are **partition and workload specific**, not global. Follow
  the per-family caps in the table below; check Slurm availability and account
  limits before each request. For the current thesis campaign, the user has set
  a hard aggregate cap of **2 GH200 GPUs total across all active allocations**
  and **2 per allocation request**. Count GPUs across jobs and nodes, including
  pending jobs that could overlap. For example, requesting 1 GH200 on gpuarm1
  plus 2 on gpuarm2 (or 2 on gpuarm1 plus 1 on gpuarm2) is forbidden because
  the combined total is 3. For this campaign, use at most 1 GH200 per node;
  the two-GPU layout is exactly one on gpuarm1 and one on gpuarm2. This cap
  does not apply to V100 or other GPU families.
- Before any GH200 request, inspect all active and pending campaign jobs with
  `squeue` and `scontrol show job`. Never request more than 2 GH200 in one job,
  combine allocations to run more than 2 simultaneously, or request a 2-GH200
  allocation on one node plus any additional GH200 on another. To avoid an idle
  gap, one successor request for 2 GH200 may be queued while the current pair
  runs only when it has an explicit Slurm dependency on the current job's
  termination, guaranteeing no allocation overlap. Do not create multiple
  pending successors or submit an unchained overlapping request. Confirm the
  dependency and aggregate requested GRES in `scontrol show job`; no more than
  2 GH200s may be allocated concurrently.

## Before allocating (after user OK) — check availability

On the **login node**, pick resources from what is actually free on each
candidate node. Before requesting a multi-node job, inspect every requested node
individually. Record CPU allocated/idle, total/free/allocated host memory, and
configured/allocated/used GPU type and count; a node being `MIXED` or having
free CPUs/RAM does not prove that the required GPU is available.

```bash
# Per-node partition, CPU, RAM, state, and configured GPU inventory
sinfo -N -p gpu-a100-80g,gpu-amd,gpu-h100-80g,gpu-h200-141g-short,gpu-h200-141g-m,gpu-grace-h200-141g,gpu-b300-288g-short \
  -o '%N %P %T %C %m %e %G'
# For each candidate node, inspect CPUAlloc/CPUTot, AllocMem/RealMemory,
# FreeMem, CfgTRES, AllocTRES, and GresUsed. Then inspect users' jobs/steps.
scontrol show node <candidate-node>
squeue -w <candidate-node> -o '%.18i %.10u %.12P %.20j %.8T %.10M %.24R'

# Queue pressure (optional)
squeue -p gpu-a100-80g,gpu-amd,gpu-h100-80g,gpu-h200-141g-short,gpu-h200-141g-m,gpu-grace-h200-141g,gpu-b300-288g-short -o '%.18i %.9P %.2t %.8N' | head
```

Check every intended node with `scontrol show node`; `sinfo`'s aggregate
partition line is not enough for CPU/RAM/GPU sizing. If available GPU usage is
not exposed in Slurm fields, inspect it only through an existing allocation or
authorized shell; never create a diagnostic `srun` step for the check. Confirm
partition `MaxTime`, account/QOS eligibility, node architecture, and the
matching environment before choosing wall-clock/resources.

### One-minute allocation wait limit and GPU-family fallback

Before every real GPU allocation attempt, inspect each candidate node's GPU,
CPU, and host-RAM availability as described above, then compare realistic
resource shapes with `srun --test-only`. The user requires the agent to spend
no more than **60 seconds waiting or polling** for an eligible allocation.
For `srun`/`salloc`, use Slurm's immediate timeout (`--immediate=60`) where
supported; never leave a blocking allocation command waiting longer. For
submitted jobs, stop polling for a start after 60 seconds. Once Slurm grants
the allocation, the training or other authorized work may run for its full
requested walltime; the one-minute limit is not a runtime limit.

If the request has not started by then, stop waiting and treat that resource
shape as unavailable for immediate work. Stop polling it and search for an
eligible alternative. Before considering each alternative, inspect every
candidate node's allocated/free CPUs, host RAM, and configured/allocated GPU
inventory. Do not repeatedly poll, submit a duplicate, or create an overlapping
replacement. This timeout does **not**
authorize `scancel` or terminate an already-pending job: follow the cancellation
authorization rule, preserve the pending job, and resolve it before requesting
an alternative if it could overlap. Compare alternatives using fresh per-node
CPU/RAM/GPU checks and `srun --test-only` forecasts.

This timer starts when a request is eligible to run. A previously approved
successor held behind an explicit `afterany`/`afterok` dependency is waiting on
its predecessor, not waiting for free resources; preserve that single chained
request and start its 60-second resource timer only after the dependency clears.

When the preferred GPU family is unavailable within the limit, search for a
compatible GPU **no lower than A100 or MI210**. Candidate families include
A100, MI210, H100, H200, GH200, and B300 when the specific node and partition
can satisfy the job. Compare predicted start time and measured end-to-end
makespan, not model names alone. MI210 requires a validated ROCm-compatible
environment; do not send the CUDA-only environment to it. NVIDIA alternatives
require their matching CUDA/PyTorch environment and representative-kernel
validation. If no eligible candidate is available, report the resource block
and continue independent work; do not silently drop below the GPU floor.
Respect the current thesis cap of two GH200s across active and pending jobs; it
does not apply to other families. Do not switch a live or dependency-queued job
to another family by submitting an overlapping replacement.

### GPU family and partition limits

The following is the Triton configuration observed on **2026-10-16** using
`sinfo` and `scontrol`; node availability is transient. GPU counts are
configured partition capacity, not a promise that those GPUs are free. Always
query `sinfo`, `squeue`, `scontrol show node`, and account/QOS before a real
request. The two-GH200 aggregate cap is specific to the current thesis
campaign; it does not limit other GPU families.

| GPU family | VRAM / CUDA capability | Triton partition(s) | Partition MaxTime / MaxNodes | Project request guidance |
|---|---|---|---|---|
| V100 | 16 or 32 GiB / 7.0 | `gpu-v100-16g`, `gpu-v100-32g`; `gpu-interactive` is 32 GiB | 5d / max one node; interactive 4h / max one node | Representative nodes: 32 CPUs/375 GiB (`gpu35`); DGX 80 CPUs/516 GiB (`dgx1`, `dgx5`). The 2026-10-16 two-trainer probe used 1 V100, 16 CPUs, and 192 GiB requested RAM; it averaged 13 FPS per trainer at `n_envs=64`. The post-run profiler query did not finish; treat this as preliminary, not a campaign throughput guarantee. Use `.venv-v100`; `.venv-x86` cu130 lacks `sm_70`. |
| A100 | 80 GiB / 8.0 | `gpu-a100-80g` | 5d / max one node | Representative node `gpu11`: 96 CPUs/515 GiB. Choose card count from benchmark and node capacity; no GH200-derived cap. |
| H100 | 80 GiB / 9.0 | `gpu-h100-80g`, `gpu-h100-80g-short` | 5d or 2d / max one node | Representative node `gpu45`: 192 CPUs/990 GiB. Match runtime and environment/kernel support to the selected partition. |
| H200 | 141 GiB / 9.0 | `gpu-h200-141g-m`, `gpu-h200-141g-short` | 8h or 2d / max one node | Representative node `gpu50`: 128 CPUs/1,980 GiB; full nodes expose up to 8 GPUs. Measure concurrency and request supported node resources. |
| H200 partitioned | 71 GiB slice (H200 3g.71gb) / 9.0 | `gpu-h200-71g-ia` | 8h / max one node | `gpu49`: 128 CPUs/1,980 GiB and up to 8 71-GiB slices alongside 4 full H200s. Treat a slice as a partitioned GPU. Other 18/35-GiB IA partitions were not configured in this snapshot. |
| GH200 | 141 GiB / 9.0 | `gpu-grace-h200-141g` | 5d / max two nodes | Current thesis-campaign user cap: **2 GH200 total across active and pending allocations**, normally one per node on two nodes. This is not the cluster's installed capacity and does not cap other GPU families. |
| B300 | 288 GiB / 10.3 | `gpu-b300-288g-short` | 2d / max one node | Representative node `gpu64`: 256 CPUs/3,000 GiB. Verify framework support and measured throughput before choosing it. |
| AMD MI100 / MI210 | 32 GiB MI100 / 64 GiB MI210, ROCm (not CUDA) | `gpu-amd` | 5d / max one node | `gpuamd1`: 16 CPUs/250 GiB, 1 MI100 + 2 MI210. Requires ROCm-compatible PyTorch/runtime; CUDA environments do not apply. |

These capability/partition values describe the configured server at the
snapshot date. Refresh them from Triton before changing allocation guidance.

### GPU count

Pick **1 or 2** GPUs from the job, not from habit. Login (`login4`: 40 CPUs,
250 GB RAM, no GPU) cannot run `n_envs=64`. A PO3 worker is budgeted at 4 GB;
64 workers OOM at 256 GB, so one 64-env train needs about **512 GB**.

This 512 GB guidance is for the default `env.data_source: frames` path. The
opt-in shared bundle path was measured separately on a GH200 (2026-10-09) with
`scripts/bench/measure_shared_ppo_memory.py`: TCN PPO at `n_envs=64` processed
10,240 steps over five iterations at about **3,811 FPS**, with PyTorch peak
allocated/reserved VRAM of **261/302 MiB**. A sampled two-trainer run used about
**2.2 GiB total** in `nvidia-smi` (about 1.09 GiB per process). An 18-trainer
projection is roughly **19–20 GiB VRAM**; budget at least **30 GiB** until an
18-trainer run confirms it. These short probes do not establish full-run host
RAM requirements, so keep the 512 GB frames-path allocation guidance. Do not
present the 18-trainer projection as a direct measurement.

| Job | GPUs | Typical request | Why |
|-----|------|-----------------|-----|
| One model (one TCN / GRU / Transformer, 20M) | **1** | Example GH200: `--gpus=gh200:1 --mem=512G --cpus-per-task=48 --time=6:00:00` or `--time=12:00:00` | GH200 example only; size CPU/RAM/time for the chosen family, partition, and measured workload. |
| Six-run matrix (overlay then baseline, three archs) | **2 GH200** | `--gpus=gh200:2 --mem=900G --cpus-per-task=128 --time=6:00:00` | Historical GH200 shape only. The GH200 campaign cap does not set V100/A100/H100/H200/B300 caps. Check [current matrix scripts](../../scripts/matrix/) before use. |
| 18-trainer full-split screen (campaign-specific) | **2 total, one per node** | GH200: `--nodes=2 --ntasks=2 --cpus-per-task=48 --gpus-per-node=gh200:1 --mem=512G --time=6:00:00` | Current active allocation shape: 18 workers across two nodes (9/node), 5 CPUs/worker, `n_envs=64`, shared bundle. Keep this distinct from two GPUs on one node; recheck cgroup memory and controller placement before reuse. |
| V100 throughput probe (2026-10-16 user-authorized) | **1 V100, 2 trainers** | Actual probe: `gpu35`, `--gres=gpu:v100:1 --mem=192G --cpus-per-task=16 --time=4:00:00` | One V100 shared by two matched throughput workers. Benchmark-only; not thesis evidence or a validated general V100 cap. |

Choose a GPU family based on availability and compatible runtime, not a global
preference. `.venv-x86` cu130 supports H100/H200 but not V100; use the separate
V100 environment only after its CUDA kernels and real GPU operation pass the
preflight below. The six-run matrix's two-GH200 shape remains specific to that
campaign.

Setup x86 env once on an x86 node: `scripts/setup/setup_venv_x86.sh`. Inside any GPU
shell: `source scripts/triton/activate_venv.sh` (picks `.venv` or `.venv-x86`).

### Time / mem / CPUs

| Knob | 1 GPU (one model) | 2 GPU (six-run) |
|------|-------------------|-----------------|
| Time | **6h or 12h.** Default `--time=6:00:00`. Use `--time=12:00:00` when the partition `MaxTime` allows it and the user asked for 12h. | **`--time=6:00:00`** |
| Mem | **`--mem=512G`** for one `n_envs=64` train. Steady use is ~300G; this feature set peaked ~473G. 256G OOMs at worker start. | **`--mem=900G`** (stay under the ~1.1 TB node). Same 900G figure for two trains on **one** H200 — see below. |
| CPUs | **`--cpus-per-task=48`** | **`--cpus-per-task=128`** (node has 144) |
| Tasks | `--ntasks=1` | `--ntasks=1` |

Example, one TCN, GH200 free, user OK (6h, or 12h if they asked):

```bash
srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=6:00:00 \
  --mem=512G --ntasks=1 --cpus-per-task=48 --pty bash
```

The historical two-GH200-on-one-node example is intentionally removed: the
current thesis campaign permits at most one GH200 on each of two nodes. Follow
the live per-node inventory and scheduler forecast before constructing any
two-node request.

`attach_or_alloc.sh` follows the same priority when it allocates (override with
`PARTITION` / `GPUS` / `TIME` / `MEM` / `CPUS` env vars).

## Historical two-trainer H200 probe (2026-10-14)

This profile documents a past probe, not a standing allocation instruction.
Apply the current per-node inventory, 60-second wait limit, A100-or-better/MI210
fallback, environment compatibility, and allocation-approval rules above.

Idea 1 (`po3_ifvg`) and Idea 2 (`distribution`) at `n_envs=64` share **one**
H200. Run `bash scripts/triton/two_trains_one_h200.sh` after the user says to
allocate. Do not probe other sizes first.

| Knob | Value | Do not try first |
|------|--------|------------------|
| GPU | `--gpus=h200:1` | `h200:2`, `gh200:2` while those GPUs are already taken |
| Mem | **`--mem=900G`** | 1024G, 1200G, 1400G (they stay PENDING) |
| CPU | **`--cpus-per-task=64`** | 96 |
| Time | **`--time=6:00:00`** | 12h on this shape (stays PENDING; `srun --test-only` can still say "now") |
| Envs | `QUANT_RL_MAX_N_ENVS=64` | Leaving the cap unset: 900G selects 128 workers and OOMs |

If a request remains pending after 60 seconds, stop polling and treat that
resource shape as unavailable for immediate work, even if Slurm shows a later
estimated start. Immediately compare eligible A100-or-better/MI210 alternatives;
do not leave a terminal or background poll running. Do not automatically
cancel the pending request or create an overlapping duplicate. Check the job
reason and per-node CPU, RAM, and GPU availability; compare alternatives with
`srun --test-only`, then follow explicit cancellation and allocation approval
rules before switching.

Second train: a new tmux window that **`ssh`s the compute node**. That shell
joins the same job (`SLURM_JOB_ID` is set). `srun --overlap` / `srun --jobid=`
fails here with `Insane message length`. Do not retry overlap.

For this thesis campaign, the approved two-GH200 layout is one GH200 on each
of two different nodes; do not request both GPUs on one node. If `AllocTRES`
shows all GH200s occupied on the intended nodes, do not submit that shape or
keep waiting on it. Apply the one-minute limit above, inspect per-node state
for eligible A100-or-better or MI210 alternatives, then follow the allocation
approval and no-overlap rules.

## Historical 18-variant matrix measurements

The profiles below record prior measurements, not current request instructions
or authorization. The current one-minute wait and alternative-selection rules
above, per-node inventory, one-GH200-per-node campaign shape, and explicit
approval requirement take precedence. Do not reproduce an old allocation shape
without a fresh forecast and approval.

The unified matrix launcher is
[`scripts/matrix/run_18variant_matrix.sh`](../../scripts/matrix/run_18variant_matrix.sh).
Select the GPU family with `GPU_KIND=gh200|h100|h200` and the allocation shape
with `MODE=1|2|3|4`:

| Mode | Allocation | Trainers | `n_envs` |
| --- | --- | ---: | ---: |
| 1 | 1 GPU, 12h, 900G | 4 on one GPU | 32 |
| 2 | 2 GPUs on one node, 6h, 900G | 2 per GPU | 32 |
| 3 | 1 GPU on each of 2 nodes, 6h, 900G per node | 2 per node | 64 |
| 4 | 1 GPU on each of 2 nodes, 6h, 600G per node | 2 per node | 32 |

The ordered subset can be selected with a whitespace-separated `VARIANTS`
value; when omitted, all entries in `scripts/variant_list.txt` are used.
The tmux entrypoint passes these options through:

```bash
MODE=2 GPU_KIND=gh200 OUT_DIR=outputs/gpu_matrix_18_mode2 \
  bash scripts/matrix/run_18variant_matrix_gh200_tmux.sh
```

Each run keeps logs under `<OUT_DIR>/logs/`: `all.log` contains the combined
launcher and Slurm output, `orchestrator.log` contains the tmux wrapper output,
and one `<variant>.log` file is written per trainer. Model artifacts and
checkpoints remain under `<OUT_DIR>/<variant>/`.

When the launcher is running inside an existing Slurm allocation, it checks
the remaining allocation time before every batch. If fewer than 4.5 hours
remain, it submits the unstarted variants with the same resource shape and
the same output/checkpoint directory, then retires the old allocation after
the replacement completes. A batch already in progress is never interrupted
by this check.

These points were measured on Triton on 2026-10-08. Treat them as historical
benchmark evidence only; they do not override the current thesis allocation
shape or the one-minute alternative-selection rule above:

- The working four-trainer shape is **one `gpu-grace-h200-141g` allocation**:
  `--gpus=gh200:2 --mem=900G --cpus-per-task=128 --time=6:00:00`.
  Run two trainers on each GPU and set `QUANT_RL_MAX_N_ENVS=32` for every
  trainer. Four `n_envs=64` trainers exceed the node's approximately 1.1 TB
  RAM; four `n_envs=32` trainers reached PPO startup successfully.
- `--exclusive` reserves the whole node. On H100 nodes this reports/reserves
  all 192 CPUs even when `--cpus-per-task` is lower. Omit `--exclusive` when
  sharing is intentional. Two independent one-GPU allocations can target
  different nodes, but each allocation may still wait for priority.
- A node can be physically mixed and still have no free GPU. Check
  `AllocTRES`, not only CPU/RAM (`gpu45` showed all four H100 GPUs allocated
  while CPU/RAM remained free).
- `srun --test-only` validates shape and gives an estimate; it does not reserve
  resources. A pending job with `Reason=Priority` is not a malformed request.
- The project's `.venv-x86` uses `torch==2.14.0+cu130`, which does not contain
  kernels for V100 compute capability 7.0. The isolated `.venv-v100` built on
  2026-10-16 uses `torch==2.14.0+cu126`, Python 3.12, SB3 2.9.0, Gymnasium
  1.3.0, and NumPy 2.2.6; `torch.cuda.get_arch_list()` includes `sm_70`, the
  detected capability is `(7, 0)`, and a 1024×1024 CUDA matmul succeeded.
  `uv pip check` reported all installed packages compatible. This verifies the
  environment and basic CUDA execution only; it does not yet establish
  representative trainer compatibility or throughput.
- H100/H200/B300 hardware may be compatible, but availability is separate from
  compatibility. Select an eligible partition and a matching environment for
  the GPU actually allocated; do not substitute a GPU family based on model
  name or presumed speed.

## Reuse (when a GPU shell already exists)

1. Confirm via `status.sh` / `find_gpu_session.sh`.
2. `tmux send-keys` into that pane; activate the **arch-matching** venv only
   **inside** the GPU bash.
3. Run train/eval/tests there. Leave the shell open when finished.

### Node architecture, environments and storage

Read [Triton environment guidance](../../docs/operations/triton_environment.md)
before environment installation or diagnosis. It covers ARM64/x86_64
selection, persistent Lustre versus ephemeral node storage, uv/venv checks,
runtime profiling, process I/O symptoms, and reproducibility. Preserve
current jobs and worktree changes.

### Mypy startup on Lustre

When `mypy --version` hangs in the architecture-matching venv, inspect
the process/I/O state and follow
[`docs/operations/triton_mypy_lustre.md`](../../docs/operations/triton_mypy_lustre.md).
Node-local staging passed a 355-file full-tree check on aarch64/GH200
(2026-10-11); x86_64 staging is documented but not yet verified.
Do not skip the full-tree type gate, alter production environments, or
allocate extra Slurm steps merely to diagnose startup latency.

## Login node vs GPU

Login-node SSH: `ls`, `cat`, `tmux capture-pane`, `squeue`, `sinfo`, edits,
`grep`, log tails, inspect-only `scripts/triton/*.sh`.

Never `source` the wrong-arch venv on login (`.venv` is aarch64-only). Use
`scripts/triton/activate_venv.sh` only inside the GPU bash.

## Train + eval

- `quant_rl.train.train_rl` already runs train **and** eval in one process.
- Prefer one invocation inside an existing (or user-approved) GPU shell.
- After train finishes, leave the GPU bash / tmux window running.

### Mandatory preflight: `srun --test-only`

Before requesting a new GPU allocation, run a Slurm dry-run on the login node with **the exact proposed partition, node, GPU type/count, CPU count, memory and walltime**:

```bash
squeue -u "$USER"
scontrol show node gpuarm1 | grep -E 'State=|CfgTRES=|AllocTRES='
srun --test-only -p gpu-grace-h200-141g -w gpuarm1 --gpus=gh200:1 --cpus-per-task=128 --mem=450G --time=12:00:00 /bin/true
```

Replace the example values with the intended request; never blindly submit the example. `--test-only` reports Slurm's **estimated schedulability/start time** and does not start a GPU training job. It is a forecast, **not a guarantee** of immediate allocation. An apparently free GPU or low `AllocTRES` alone does not prove that Slurm can start the request; reservations, priorities, CPU, memory and duration matter.

If the forecast is delayed, test realistic CPU/memory/walltime alternatives with `srun --test-only` **before** any real `srun` or `sbatch`. Test a minimal resource shape if necessary to distinguish GPU scheduling constraints from excessive CPU/RAM requests. For example, when even one GH200, one CPU and 1 GiB with the same walltime is delayed, reducing CPU/RAM cannot by itself guarantee an immediate start. Check `squeue` and `scontrol` for real jobs; dry-run job IDs are not live allocations. Do not repeatedly submit and cancel real jobs to probe availability.


### Compare priority and GPU alternatives before waiting

For CPU-heavy PPO campaigns, minimize estimated total completion time:
scheduler wait plus measured training runtime. Do not select GH200 by default
if another supported family can start sooner and meets the workload's CPU, RAM,
VRAM, CUDA, architecture, environment and walltime requirements.

Inspect account eligibility with `sacctmgr show assoc user=$USER
format=Account,Partition,QOS` and partition priority with
`scontrol show partition <partition>`. Priority is not a reservation. Compare
realistic resource shapes with login-node `srun --test-only` before any real
allocation, while respecting each partition's maximum walltime:

```bash
srun --test-only -p gpu-a100-80g --gpus=a100:1 --cpus-per-task=32 --mem=180G --time=02:00:00 /bin/true
srun --test-only -p gpu-h100-80g --gpus=h100:1 --cpus-per-task=32 --mem=180G --time=02:00:00 /bin/true
srun --test-only -p gpu-h200-141g-short --gpus=h200:1 --cpus-per-task=32 --mem=180G --time=02:00:00 /bin/true
srun --test-only -p gpu-b300-288g-short --gpus=b300:1 --cpus-per-task=32 --mem=180G --time=02:00:00 /bin/true
srun --test-only -p gpu-grace-h200-141g --gpus=gh200:1 --cpus-per-task=32 --mem=180G --time=02:00:00 /bin/true
```

Use `scontrol show node <node>` to distinguish configured GPUs (`CfgTRES`)
from allocated GPUs (`AllocTRES`). A `MIXED` node state does not mean a GPU is
free. Treat dry-run start estimates as forecasts, not reservations. Verify the
environment on the requested architecture and use a representative training
benchmark before increasing trainer concurrency. Never submit a real job only
because a dry-run predicted a short wait.
