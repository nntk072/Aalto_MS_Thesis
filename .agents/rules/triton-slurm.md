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
- More than **one GPU** per allocation unless the user asks for multi-GPU.

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

### GPU priority (1 GPU only)

| Priority | When | Partition (typical) | GRES | CPU arch / venv |
|----------|------|---------------------|------|-----------------|
| **1 — H200** | H200 partition has idle/usable capacity | `gpu-h200-141g-short` (preferred) or `gpu-h200-141g-m` | `--gpus=h200:1` | **x86_64** — use an x86 venv (not the GH200 aarch64 `.venv`) |
| **2 — GH200** | No usable H200 (full queue, no access, or user asks) | `gpu-grace-h200-141g` | `--gpus=gh200:1` | **aarch64** — project `.venv` on scratch |

Always **`--gpus=<type>:1`** (one GPU). Do not request 2+.

### Time / mem / CPUs

| Knob | Rule |
|------|------|
| Time | Prefer **`--time=12:00:00`** when the partition `MaxTime` allows it; otherwise **`--time=6:00:00`** (or the partition max if lower, e.g. `gpu-h200-141g-m` is 8h so 6h is fine). |
| Mem | Start **`--mem=128G`**. Raise only after OOM (256G / 512G for high `n_envs`). Stay under partition memory. |
| CPUs | Start **`--cpus-per-task=8`**. Raise for high `n_envs` (e.g. 48) only if needed; check node CPU count via `sinfo`. |
| Tasks | `--ntasks=1` |

Example after checks + user OK (H200 available, MaxTime ≥ 12h):

```bash
srun --partition=gpu-h200-141g-short --gpus=h200:1 --time=12:00:00 \
  --mem=128G --ntasks=1 --cpus-per-task=8 --pty bash
```

Fallback when only GH200 is usable:

```bash
srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=12:00:00 \
  --mem=128G --ntasks=1 --cpus-per-task=8 --pty bash
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

Never `source` the wrong-arch `.venv` on login (GH200 `.venv` is aarch64).

## Train + eval

- `quant_rl.train.train_rl` already runs train **and** eval in one process.
- Prefer one invocation inside an existing (or user-approved) GPU shell.
- After train finishes, leave the GPU bash / tmux window running.
