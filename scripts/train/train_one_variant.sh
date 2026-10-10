#!/usr/bin/env bash
# Train one ablation variant. Every setting (algo, arch, strategy, reward mode,
# feature profile, steps, seeds) comes from quant_rl/train/variant_resolver.py,
# which reads config/experiments.yaml. Nothing is hard-coded here.
#
# Usage:
#   VARIANT=ladder_a2_tp scripts/train/train_one_variant.sh
#   VARIANT=ladder_a2_tp SEED=43 scripts/train/train_one_variant.sh   # one seed
#   DRY_RUN=1 VARIANT=ablation_sac_agent scripts/train/train_one_variant.sh
#
# Environment:
#   VARIANT   (required) name from config/experiments.yaml
#   SEED      run only this seed (default: every seed declared for the variant)
#   SEEDS     override the declared seed list, e.g. "42 43 44"
#   STEPS     override total timesteps (default: declared in experiments.yaml)
#   VAE_PATH  required when the variant declares use_vae: 1
#   GPU_ID    CUDA device index (default 0)
#   OUT_DIR   output root (default outputs/gpu_matrix)
#   DRY_RUN   1 = print the resolved commands and exit without training
set -euo pipefail

REPO="${TRITON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"

VARIANT="${VARIANT:-}"
if [[ -z "$VARIANT" ]]; then
  echo "ERROR: VARIANT is required (see config/experiments.yaml)" >&2
  exit 1
fi

SEED="${SEED:-}"
SEEDS="${SEEDS:-}"
STEPS="${STEPS:-}"
GPU_ID="${GPU_ID:-0}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/gpu_matrix}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
VAE_PATH="${VAE_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  PYTHON="${PYTHON:-python3}"
else
  if [[ -f "$REPO/scripts/triton/activate_venv.sh" ]]; then
    # shellcheck disable=SC1091
    source "$REPO/scripts/triton/activate_venv.sh"
  fi
  PYTHON="${PYTHON:-python}"
fi

RESOLVE_OPTS=()
[[ -n "$STEPS" ]] && RESOLVE_OPTS+=(--steps "$STEPS")
[[ -n "$SEEDS" ]] && RESOLVE_OPTS+=(--seeds "$SEEDS")
[[ -n "$VAE_PATH" ]] && RESOLVE_OPTS+=(--vae-path "$VAE_PATH")

resolver() {
  PYTHONPATH="$REPO" "$PYTHON" -m quant_rl.train.variant_resolver "$@"
}

# Resolve once; a failure here (unknown variant, VAE missing, bad config) stops the job.
META="$(resolver meta "$VARIANT" "${RESOLVE_OPTS[@]}")"
R_ALGO=""; R_ARCH=""; R_USE_VAE=""; R_STEPS=""; R_SEEDS=""
R_STRATEGY=""; R_FEATURE_PROFILE=""; R_REWARD_MODE=""
while IFS='=' read -r key value; do
  case "$key" in
    ALGO) R_ALGO="$value" ;;
    ARCH) R_ARCH="$value" ;;
    USE_VAE) R_USE_VAE="$value" ;;
    STEPS) R_STEPS="$value" ;;
    SEEDS) R_SEEDS="$value" ;;
    STRATEGY) R_STRATEGY="$value" ;;
    FEATURE_PROFILE) R_FEATURE_PROFILE="$value" ;;
    REWARD_MODE) R_REWARD_MODE="$value" ;;
  esac
done <<< "$META"

if [[ -n "$SEED" ]]; then
  SEED_LIST=("$SEED")
else
  read -r -a SEED_LIST <<< "$R_SEEDS"
fi

echo "=== variant=$VARIANT algo=$R_ALGO arch=$R_ARCH strategy=$R_STRATEGY reward=$R_REWARD_MODE" \
     "features=$R_FEATURE_PROFILE steps=$R_STEPS use_vae=$R_USE_VAE seeds=${SEED_LIST[*]} ==="

run_seed() {
  local seed="$1"
  local tag="${VARIANT}__seed${seed}"
  local run_out="$OUT_DIR/$tag"
  local log="$LOG_DIR/$tag.log"
  local done_marker="$OUT_DIR/$tag.done"
  local failed_marker="$OUT_DIR/$tag.failed"

  local args_txt
  args_txt="$(resolver args "$VARIANT" --seed "$seed" "${RESOLVE_OPTS[@]}")"
  local -a train_args=()
  mapfile -t train_args <<< "$args_txt"
  if [[ -n "${QUANT_RL_DATA_SOURCE:-}" ]]; then
    case "$QUANT_RL_DATA_SOURCE" in
      frames|bundle) train_args+=("env.data_source=$QUANT_RL_DATA_SOURCE") ;;
      *) echo "ERROR: invalid QUANT_RL_DATA_SOURCE=$QUANT_RL_DATA_SOURCE" >&2; return 2 ;;
    esac
  fi

  mkdir -p "$run_out" "$LOG_DIR"
  resolver manifest "$VARIANT" --seed "$seed" --out "$run_out/resolved_config.json" \
    "${RESOLVE_OPTS[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN $tag: $PYTHON -m quant_rl.train.train_rl --out $run_out ${train_args[*]}"
    return 0
  fi
  if [[ -f "$done_marker" ]]; then
    echo "SKIP $tag (already done)"
    return 0
  fi

  rm -f "$failed_marker"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export QUANT_RL_MAX_N_ENVS="${QUANT_RL_MAX_N_ENVS:-64}"
  export PYTHONUNBUFFERED=1

  local status=0
  set +e
  "$PYTHON" -u -m quant_rl.train.train_rl "${train_args[@]}" --out "$run_out" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -eq 0 ]]; then
    rm -f "$failed_marker"
    touch "$done_marker"
  else
    touch "$failed_marker"
  fi
  echo "=== $tag exit=$status ==="
  return "$status"
}

overall=0
for seed in "${SEED_LIST[@]}"; do
  run_seed "$seed" || overall=1
done
exit "$overall"
