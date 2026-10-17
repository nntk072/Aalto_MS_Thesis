# Triton architecture, filesystems and Python environments

This guide applies to agents working on Aalto Triton in
`/scratch/work/nguyenl37/Aalto_MS_Thesis`. Consult the
[Slurm agent rules](../../.agents/rules/triton-slurm.md) before allocating,
cancelling or adding compute steps.

## Identify the actual compute environment

Connect using the configured SSH host `code.triton.aalto.fi`. Login nodes may
be x86_64 even when the active GPU allocation is ARM64. Check the *compute*
shell: `hostname`, `uname -m`, `echo "$SLURM_JOB_ID"`, and `nvidia-smi`.
Do not infer node architecture from the SSH gateway.

| Architecture | Observed example | Project environment | Installation |
| --- | --- | --- | --- |
| ARM64 / `aarch64` | `gpuarm1`, NVIDIA GH200 | `.venv` | `uv sync --locked --extra dev` with ARM64 uv |
| AMD64 / `x86_64` | H100/H200 compute nodes | `.venv-x86` | `scripts/setup/setup_venv_x86.sh` on x86_64 |
| AMD64 / `x86_64` | V100 (CC 7.0) | `.venv-v100` | Separate environment; torch 2.14.0+cu126, Python 3.12, project lock dependencies excluding its cu130 CUDA packages. Verify `sm_70`, CC `(7, 0)`, and a real CUDA op before use. |
| AMD64 / `x86_64` | MI210 (`gfx90a`) | `/scratch/work/nguyenl37/.venv-mi210` | `scripts/setup/setup_venv_mi210.sh`; observed Triton ROCm 6.3.1, Python 3.12.14, torch 2.7.1+rocm6.3. Keep separate from the CUDA 13.0 project lock; validate the runtime and a real device operation before use. |

Activate with `source scripts/triton/activate_venv.sh` inside the allocation.
The selector uses `.venv-mi210` on x86_64 `gpu-amd` allocations and `.venv-x86`
on other x86_64 nodes; GH200 uses `.venv`. Inspect `SLURM_JOB_PARTITION`,
`command -v python`, `python -c 'import sys; print(sys.prefix)'`, and
`python --version` to verify the selected environment. A ROCm device smoke is
still required before claiming the MI210 runtime works.
Never execute ARM64 venv entrypoints from an x86_64 login shell or reuse
compiled packages across architectures. GPU types, RAM, partitions, and
availability change; inspect current Slurm state rather than hardcoding
resource assumptions.

The x86 `.venv-x86` is built with torch 2.14.0+cu130 and does not include V100
(`sm_70`) kernels. The V100 environment is isolated from it and uses the
CUDA 12.6 wheel that includes `sm_70`. Its creation must not edit `pyproject.toml`
or `uv.lock`; install application dependencies from the locked export while
excluding the torch/cu130 CUDA runtime packages, then install `torch==2.14.0`
from the official cu126 index. Re-run the compatibility checks after recreating
the environment. Do not claim V100 training support from environment creation
alone; a representative trainer smoke run is still required.

The MI210 environment is isolated from `.venv-x86`: the project lock selects
CUDA 13.0 packages, while the observed MI210 node exposes ROCm 6.3.1. The setup
script keeps the locked non-accelerator dependencies and installs the official
PyTorch 2.7.1 ROCm 6.3 wheel. On `gpuamd1` (Slurm job `21007645`), a 256×256
MI210 matrix multiply succeeded, and the semantic-head/hybrid PPO update and
save/load checks passed with this environment. This establishes a bounded test
runtime, not parity with the locked CUDA 13.0 training stack or qualification
for a research campaign. On 2026-10-17, job `21012389` repeated a real MI210 GPU
operation and passed the full `pytest tests/ -q` suite (1,594 tests, 40
warnings). A source-snapshot validation on the same allocation (step 3, HEAD
`599741fe710fa8b342ef3797448132413c07e4db`) then passed full-tree Ruff
format/lint, full-tree Mypy (458 files), a real MI210 matrix operation, and all
1,603 tests with 38 warnings. Its log is
`outputs/gpu_ci_gates/q0_q7_current_tree_20261017_job21012389/current_tree_validation.log`.
This historical result does not validate later dirty-tree changes.
This adds cross-runtime test evidence but still does not establish full trainer
parity or throughput. An end-to-end trainer and resource/throughput benchmark
remain required before MI210 training use.

## Storage characteristics

- **Lustre scratch** (`/scratch/work/nguyenl37`): persistent project,
  source, venvs, wheel caches, checkpoints, logs and reproducible reports.
  Distributed filesystem metadata and small-file imports can be slow even when
  space and inode quotas are sufficient.
- **Node-local `/tmp`**: ephemeral fast storage for staged Python runtimes,
  command-line tool modules, caches and temporary experiments. Its contents
  may disappear when the allocation ends or a different node is used.
  Never keep the only copy of a checkpoint, dataset or report there.
- **`/home`**: separate space/quota; avoid placing large CUDA wheels there
  without checking quota.
- **tmux** persists terminal sessions but does not guarantee that their Slurm
  allocation remains active. Save logs to scratch with job ID, architecture,
  command, seed, configuration, timestamps and exit status.

Before copying large environments, check `df -h /tmp /scratch`, environment
sizes with `du -sh`, and actual user quota. A configured Slurm RAM request
is not an observed memory peak. Benchmark host RAM, CPU, GPU utilization and
GPU memory using monitored processes/cgroups, stating whether figures are
peaks, snapshots, or limits.

## Installation and verification

Run installation on the target architecture in an authorized compute shell.
For reproducible installs, select the matching interpreter and uv binary,
inspect `uv.lock`, and use `UV_PROJECT_ENVIRONMENT` for the intended
absolute environment path. For ARM64, that is `$PWD/.venv`; for x86_64,
`$PWD/.venv-x86`. On an authorized H100/H200 compute node, run
`scripts/setup/setup_venv_x86.sh`; it uses the locked command
`UV_PROJECT_ENVIRONMENT="$PWD/.venv-x86" uv sync --locked --extra dev`.
Do not use a project `.venv` symlink whose target is a different checkout.

A uv-created venv may have no `pip` module: `python -m pip` failing is
not proof the environment is broken. Prefer `uv pip check --python
<venv>/bin/python` and `uv sync --check`, together with direct import and
runtime checks. Verify in increasing depth: Python startup, imports and CLI
tools (ruff/mypy/pytest), PyTorch CUDA detection, a small real GPU operation,
relevant tests, then required full-tree CI checks.

**Measured ARM64 case (2026-10-11):** Python 3.12.15, uv 0.13.0,
148 installed packages, torch 2.14.0+cu130, SB3 2.9.0, and successful CUDA
matrix multiplication on GH200. The NVIDIA `nvidia-cusparselt-cu13==0.8.1`
wheel had internal `manylinux2014_sbsa` metadata despite an aarch64 wheel
filename and aarch64 ELF library; `uv pip check` reported a platform
mismatch. Thus CUDA execution worked, while dependency metadata validation
was **not fully clean**. Do not silently ignore or automatically change
locked dependencies to hide this result.

## Diagnose hangs and failed installs

1. Verify actual node architecture, interpreter path, selected venv, and exact
   command; inspect existing logs before restarting anything.
2. Use `ps -u "$USER" -o pid,stat,wchan:24,etime,pcpu,args` to inspect stuck
   processes. `folio_wait_bit_common` and `cl_sync_io_wait` can indicate
   filesystem I/O waits, but are evidence rather than proof of root cause.
3. Separate resolution, download, creation and installation phases:
   `uv sync --locked --extra dev --dry-run` identifies pending operations.
   Inspect uv processes and environment locks before any new sync; do not
   launch competing installers.
4. Compare Python startup and representative imports from Lustre with a
   private node-local staging directory. `timeout` exit 124 indicates a
   timeout; an empty log may reflect output buffering. Preserve the existing
   environment and artifacts until a replacement is verified.
5. Stage only matching-architecture runtime components temporarily. Do not
   overwrite production environments, change trading configurations, or
   cancel/reallocate jobs as troubleshooting shortcuts.

**Observed ARM64 incident:** an incomplete `.venv` initially had empty
site-packages after interrupted sync attempts. A later locked sync installed
148 packages. Direct `mypy --version` subsequently stalled during imports
from Lustre; node-local Python and mypy ran the full `mypy .` check:
**355 source files, zero errors**. The direct startup issue was not repaired.
See [Mypy-specific workaround](triton_mypy_lustre.md). The generic x86_64
procedure is documented but **not yet experimentally validated**.

## Agent recording checklist

Record SSH host, compute node, architecture, Slurm allocation and remaining
time, GPU model, chosen venv, Python/uv versions, git SHA and dirty state,
lockfile and config, seed, exact command, logs, exit code, all resource
measurements, and their provenance. Retain authoritative results on scratch.
Reuse existing authorized tmux/Slurm shells under repository rules, preserve
unrelated edits and workloads, and distinguish ARM64 observations from
unverified x86_64 hypotheses.
