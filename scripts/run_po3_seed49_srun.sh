#!/bin/bash
set -euo pipefail
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
mkdir -p outputs
exec srun --partition=gpu-grace-h200-141g --gpus=gh200:1 --time=6:00:00 --mem=48G --ntasks=1 --cpus-per-task=8 --chdir=/scratch/work/nguyenl37/Aalto_MS_Thesis bash /scratch/work/nguyenl37/Aalto_MS_Thesis/scripts/launch_po3_seed49.sh 2>&1 | tee /scratch/work/nguyenl37/Aalto_MS_Thesis/outputs/po3_seed49_launch.log