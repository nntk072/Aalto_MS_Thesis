# Triton tmux / Slurm helpers
#
# Run on the **login node** (SSH). Agents: reuse existing GPU panes first;
# call attach_or_alloc / srun only after the user confirms in-chat.
# Prefer one long allocation (12h if MaxTime allows, else 6h), never bursts
# of short (&lt;~120s) jobs. Always 1 GPU.
#
# GPU priority: H200 (gpu-h200-141g-short, --gpus=h200:1) when available,
# else GH200 (gpu-grace-h200-141g, --gpus=gh200:1). Check sinfo first.
#
#   bash scripts/triton/status.sh           # squeue + tmux overview
#   bash scripts/triton/find_gpu_session.sh # exit 0 if reusable GPU job/session
#   bash scripts/triton/attach_or_alloc.sh  # reuse, or ONE long srun (needs OK)
#   bash scripts/triton/two_trains_one_h200.sh  # 1xH200 900G 6h, two n_envs=64 (needs OK)
#
# Defaults / overrides:
#   MEM=128G CPUS=8 TIME=auto(12h|6h) PARTITION/GPUS=auto(H200 then GH200)
#
# After training finishes: leave the srun bash and tmux window open.
# Do not scancel unless the user explicitly says to cancel that job.
# Retrain or deleting outputs is not that permission. Do not pass REPLACE=1.
#
# Agent policy: see `.agents/rules/triton-slurm.md`.
