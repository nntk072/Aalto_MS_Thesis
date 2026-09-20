#!/bin/bash
set -euo pipefail
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
mkdir -p outputs
# Year-episode 20M retrain: 64 envs need ~512G Slurm mem + 48 CPUs.
exec srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=6:00:00 \
  --mem=512G --ntasks=1 --cpus-per-task=48 \
  --chdir=/scratch/work/nguyenl37/Aalto_MS_Thesis \
  bash /scratch/work/nguyenl37/Aalto_MS_Thesis/scripts/launch_po3_seed50.sh \
  2>&1 | tee /scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/po3_seed50_year_episode_launch.log
