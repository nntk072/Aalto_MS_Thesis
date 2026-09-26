# Triton / Slurm (GPU) — agent rules

Aalto Triton cluster. **Canonical workdir:** `/scratch/work/nguyenl37/Aalto_MS_Thesis`.

Helper scripts: [`scripts/triton/`](../../scripts/triton/) (`status.sh`, `find_gpu_session.sh`, `attach_or_alloc.sh`).

## WorkDir

- All training, eval, and agent edits for GPU runs must use
  `/scratch/work/nguyenl37/Aalto_MS_Thesis` (real scratch path).
- A symlink `~/Aalto_MS_Thesis` → that scratch path is OK; always `cd` to the
  scratch path (or resolve the symlink) before any GPU-side command.
- Do **not** treat a WSL or laptop checkout as the live Triton tree.

## Hard rule: `srun` only with user confirmation

Agents **may** run `srun` / `attach_or_alloc.sh`, but **only after the user
explicitly permits it in this chat** (e.g. “ok to srun”, “allocate a GPU”,
“start the job”). Do **not** self-start allocations.

Triton emails when many jobs last &lt; ~120s. Prefer one long shell (hours),
never a burst of short diagnostic jobs.

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

- Any new `srun` (interactive or batch).
- `bash scripts/triton/attach_or_alloc.sh` when it would start a new alloc.
- `scancel` / replacing a running job.
- Extra GPU jobs for “quick” pytest / ruff / mypy / CUDA smoke when no
  reusable shell exists (prefer asking for one long alloc, or login-only
  x86-safe tools).

### Still forbidden

- Loops / bursts of short jobs or job steps (&lt; ~120s each).
- `scancel -u $USER` unless the user explicitly requests a full wipe.
- Using `srun` for login-node checks (`ls`, log tails, plain `grep`).
- More than **two GPUs** per allocation. One model uses 1 GPU. The six-run
  encoder matrix uses 2. Do not request 3+.

## Before allocating (after user OK) — check availability

On the **login node**, pick resources from what is actually free. Do not
hard-code GH200 when H200 is idle.

```bash
# Partition / GRES / MaxTime / node states
sinfo -p gpu-h200-141g-short,gpu-h200-141g-m,gpu-grace-h200-141g \
  -o '%P %a %D %T %G %l %m %c'

# Queue pressure (optional)
squeue -p gpu-h200-141g-short,gpu-h200-141g-m,gpu-grace-h200-141g -o '%.18i %.9P %.2t %.8N' | head
```

Prefer partitions with **idle** or lightly **mixed** nodes and short queues.
Confirm `MaxTime` before choosing wall-clock.

### GPU count

Pick **1 or 2** GPUs from the job, not from habit. Login (`login4`: 40 CPUs,
250 GB RAM, no GPU) cannot run `n_envs=64`. A PO3 worker is budgeted at 4 GB;
64 workers OOM at 256 GB, so one 64-env train needs about **512 GB**.

| Job | GPUs | Typical request | Why |
|-----|------|-----------------|-----|
| One model (one TCN / GRU / Transformer, 20M) | **1** | GH200: `--gpus=gh200:1 --mem=512G --cpus-per-task=48 --time=6:00:00` or `--time=12:00:00` | One process, one 141 GB GPU. 6h is the default 20M train. 12h when `MaxTime` allows and the user asks for the longer shell. |
| Six-run matrix (overlay then baseline, three archs) | **2** | GH200: `--gpus=gh200:2 --mem=900G --cpus-per-task=128 --time=6:00:00` | Same shape as [`scripts/run_6run_matrix_tmux.sh`](../../scripts/run_6run_matrix_tmux.sh). The node has 2 GPUs and ~1.1 TB. |

H200 (`gpu-h200-141g-short`, `.venv-x86`) is still preferred for a **1-GPU** job
when that partition has idle capacity. The six-run matrix stays on GH200
(`gpu-grace-h200-141g`, `.venv`) because that node is 2 GPUs and the project
env matches it. Do not request 3+ GPUs.

Setup x86 env once on an x86 node: `scripts/setup_venv_x86.sh`. Inside any GPU
shell: `source scripts/triton/activate_venv.sh` (picks `.venv` or `.venv-x86`).

### Time / mem / CPUs

| Knob | 1 GPU (one model) | 2 GPU (six-run) |
|------|-------------------|-----------------|
| Time | **6h or 12h.** Default `--time=6:00:00`. Use `--time=12:00:00` when the partition `MaxTime` allows it and the user asked for 12h. | **`--time=6:00:00`** |
| Mem | **`--mem=512G`** for `n_envs=64`. Do not start a 64-env PO3 train at 128G. | **`--mem=900G`** (stay under the ~1.1 TB node) |
| CPUs | **`--cpus-per-task=48`** | **`--cpus-per-task=128`** (node has 144) |
| Tasks | `--ntasks=1` | `--ntasks=1` |

Example, one TCN, GH200 free, user OK (6h, or 12h if they asked):

```bash
srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=6:00:00 \
  --mem=512G --ntasks=1 --cpus-per-task=48 --pty bash
```

Example, six-run matrix, user OK:

```bash
srun --partition=gpu-grace-h200-141g --gpus=gh200:2 --time=6:00:00 \
  --mem=900G --ntasks=1 --cpus-per-task=128 --pty bash
```

`attach_or_alloc.sh` follows the same priority when it allocates (override with
`PARTITION` / `GPUS` / `TIME` / `MEM` / `CPUS` env vars).

## Reuse (when a GPU shell already exists)

1. Confirm via `status.sh` / `find_gpu_session.sh`.
2. `tmux send-keys` into that pane; activate the **arch-matching** venv only
   **inside** the GPU bash.
3. Run train/eval/tests there. Leave the shell open when finished.

## Login node vs GPU

Login-node SSH: `ls`, `cat`, `tmux capture-pane`, `squeue`, `sinfo`, edits,
`grep`, log tails, inspect-only `scripts/triton/*.sh`.

Never `source` the wrong-arch venv on login (`.venv` is aarch64-only). Use
`scripts/triton/activate_venv.sh` only inside the GPU bash.

## Train + eval

- `quant_rl.train.train_rl` already runs train **and** eval in one process.
- Prefer one invocation inside an existing (or user-approved) GPU shell.
- After train finishes, leave the GPU bash / tmux window running.
