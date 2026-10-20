# Triton operations and runtime verification

For compute architecture, venv provisioning, storage characteristics,
resource observations and installation troubleshooting, see
[Triton environment guidance](triton_environment.md).

If `mypy --version` stalls inside an allocated GPU shell, see
[the architecture-aware Lustre Mypy workaround](triton_mypy_lustre.md).
The ARM64 workaround was verified on 2026-10-11; the x86_64 equivalent
needs validation on an x86 compute node.

## Historical GPU and RAM guidance

The [GPU campaign plan](../training/triton_gpu_training_agent_plan.md)
documents the 18-variant, five-seed proposal, allocation-aware scheduling,
durable per-attempt logging and resume criteria. The
[RAM optimization plan](../training/triton_ram_optimization_plan.md)
records experimental bundle/frames memory probes; short-slice memory samples
are not production PPO peak-memory or scientific parity evidence.
Connect to `code.triton.aalto.fi` and enter
`/scratch/work/nguyenl37/Aalto_MS_Thesis`. Check `git status --short --branch`
and `git rev-parse HEAD` before running experiments. Do not assume SSH shell
architecture matches allocated GPU node architecture.

Follow [docs/operations/triton/README.md](../../docs/operations/triton/README.md) and
[Slurm agent rules](../../.agents/rules/triton-slurm.md) for resource selection.
Use `bash scripts/compute/slurm/status.sh` for a status overview and
`bash scripts/compute/slurm/find_gpu_session.sh` to discover a reusable session.
`attach_or_alloc.sh` and `two_trains_one_h200.sh` can request resources;
running them requires explicit allocation authorization under repository rules.
The helper documentation prefers one H200 or GH200 allocation, and requires
checking `sinfo` rather than assuming a partition is available.

The architecture-specific environments are `.venv` (ARM64/aarch64, including
`gpuarm` GH200 nodes) and `.venv-x86` (x86_64 compute nodes such as H100/H200).
Never use `.venv-x86` on `gpuarm`; the setup scripts live under `tools/`.
Use `scripts/compute/slurm/activate_venv.sh` **inside the correct allocated shell**.
Do not use `PAPER_TRADING=false` for HPC tests or any documentation check.

For reproducible experiments, save Slurm job ID, node type, environment,
`git SHA`, resolved config, seed, launch command, checkpoint and run-log paths.
Check logs and completion artifacts before declaring success; a zero exit
status alone does not prove a valid final PPO checkpoint. See
[reproducibility](../reproducibility.md).
