#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${TRITON_REPO:-/scratch/work/nguyenl37/Aalto_MS_Thesis}"
JOB_ID="${SLURM_JOB_ID:?run inside the approved P3 allocation}"
OUT_DIR="${P3_OUT_DIR:-$REPO/outputs/thesis_validation/p3_recovery_${JOB_ID}}"
# Use the existing full-PO3 profile so the smoke exercises supplied native
# M3-D1 context through training, checkpoint restore, and deterministic replay.
VARIANT="ladder_a0_flat_po3"
SEED=50
STEPS=16384

cd "$REPO"
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing P3 evidence: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR/logs"
exec > >(tee -a "$OUT_DIR/logs/p3_recovery.log") 2>&1

source "$REPO/scripts/compute/slurm/activate_venv.sh"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export QUANT_RL_MAX_N_ENVS=64

RUN_DIR="${P3_REUSE_RUN_DIR:-}"
TRAINER_EXIT_CODE=0
if [[ -z "$RUN_DIR" ]]; then
  echo "P3 smoke job=$JOB_ID variant=$VARIANT seed=$SEED steps=$STEPS output=$OUT_DIR"
  set +e
  VARIANT="$VARIANT" \
    SEED="$SEED" \
    STEPS="$STEPS" \
    OUT_DIR="$OUT_DIR/training" \
    LOG_DIR="$OUT_DIR/logs" \
    QUANT_RL_DATA_SOURCE=bundle \
    QUANT_RL_FAST_REPORTS=1 \
    bash "$REPO/scripts/research/frozen-launches/train_one_variant.sh"
  TRAINER_EXIT_CODE=$?
  set -e

  mapfile -t manifests < <(find "$OUT_DIR/training" -name run_manifest.json -type f -print)
  if [[ "${#manifests[@]}" -ne 1 ]]; then
    echo "Expected exactly one completed run manifest; found ${#manifests[@]}" >&2
    exit 1
  fi
  RUN_DIR="$(dirname "${manifests[0]}")"
  if [[ "$TRAINER_EXIT_CODE" -ne 0 ]] && ! grep -Fq \
    "TRAIN_SURVIVAL_FAILED: train-year risk gate failed" "$OUT_DIR/logs/p3_recovery.log"; then
    echo "P3 training exited $TRAINER_EXIT_CODE for a reason other than the strategy survival gate" >&2
    exit "$TRAINER_EXIT_CODE"
  fi
  if [[ "$TRAINER_EXIT_CODE" -ne 0 ]]; then
    echo "P3 recovery smoke continuing after the strategy-survival failure; checkpoint recovery is evaluated separately."
  fi
else
  echo "P3 smoke job=$JOB_ID reusing completed checkpoint run=$RUN_DIR"
  if grep -Fq "TRAIN_SURVIVAL_FAILED: train-year risk gate failed" \
    "$RUN_DIR/../../../logs/p3_recovery.log"; then
    TRAINER_EXIT_CODE=1
  fi
fi

if [[ ! -s "$RUN_DIR/run_manifest.json" ]]; then
  echo "P3 run manifest is missing: $RUN_DIR/run_manifest.json" >&2
  exit 1
fi
CHECKPOINT="$RUN_DIR/model/ppo_final.zip"
if [[ ! -s "$CHECKPOINT" ]]; then
  echo "P3 smoke checkpoint is missing: $CHECKPOINT" >&2
  exit 1
fi

# Verify the resolved training config, not only the variant's requested base
# profile. This makes the M1-only LTF and supplied M3-D1 context an explicit P3
# acceptance condition.
python - "$RUN_DIR/config.yaml" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
features = config.features
expected_htfs = ["M3", "M5", "M10", "M15", "M30", "H1", "H4", "D1"]
if str(features.po3_ltf).upper() != "M1":
    raise SystemExit(f"P3 expected M1 as the sole PO3 LTF; found {features.po3_ltf!r}")
actual_htfs = [str(value).upper() for value in features.htf_timeframes]
if actual_htfs != expected_htfs:
    raise SystemExit(f"P3 native HTF list mismatch: {actual_htfs!r} != {expected_htfs!r}")
print(f"P3 resolved profile: po3_ltf=M1 htf_timeframes={','.join(actual_htfs)}")
PY

# Refuse expensive replays when the training run's core source identity no
# longer matches the current evaluation source.
python - "$REPO" "$RUN_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

repo, run_dir = map(Path, sys.argv[1:])
manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
required = {
    "config/features/full_po3_mtf.yaml": (),
    "market_data/loading/bars.py": ("quant_rl/data/loader.py",),
    "market_data/loading/pipeline.py": ("quant_rl/data/pipeline.py",),
    "market_data/loading/ticks.py": ("quant_rl/data/ticks.py",),
    "simulation/environment/env.py": ("simulation/envs/trading_env.py", "quant_rl/envs/trading_env.py"),
    "features/po3/state.py": ("quant_rl/features/po3_state.py",),
    "evaluation/reanalysis/runner.py": ("quant_rl/eval/eval_run.py",),
    "evaluation/reporting/run_manifest.py": ("quant_rl/eval/run_manifest.py",),
    "config/base/default.yaml": ("quant_rl/config/default.yaml",),
}
saved = {
    item["path"]: item.get("sha256")
    for item in manifest.get("source", {}).get("dirty_files", [])
}
changed = []
for path, aliases in required.items():
    current_hash = hashlib.sha256((repo / path).read_bytes()).hexdigest()
    saved_hash = next((saved[key] for key in (path, *aliases) if saved.get(key)), None)
    if saved_hash != current_hash:
        changed.append(path)
if changed:
    raise SystemExit(f"P3 checkpoint source identity is stale: {changed}")
checkpoint = run_dir / "model/ppo_final.zip"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if manifest.get("checkpoint", {}).get("sha256") != digest:
    raise SystemExit("P3 checkpoint hash does not match its run manifest")
print("P3 checkpoint and training source identity validated before replay.")
PY

python -m evaluation.reanalysis.runner \
  --run "$RUN_DIR" \
  --output "$OUT_DIR/reanalysis" \
  --max-charts 0 \
  --no-save-html \
  --allow-historical

# Repeat the exact checkpoint replay under another interpreter hash seed. This
# turns the existing runtime-provenance check into direct byte-level evidence.
env PYTHONHASHSEED=73 PYTHONUNBUFFERED=1 python -m evaluation.reanalysis.runner \
  --run "$RUN_DIR" \
  --output "$OUT_DIR/reanalysis_hashseed73" \
  --max-charts 0 \
  --no-save-html \
  --allow-historical

python scripts/research/thesis/evidence/compare_deterministic_reanalysis.py \
  "$OUT_DIR/reanalysis" \
  "$OUT_DIR/reanalysis_hashseed73" \
  --output "$OUT_DIR/p3_hashseed_repeatability.json"

python - "$REPO" "$RUN_DIR" "$OUT_DIR/reanalysis" "$OUT_DIR/reanalysis_hashseed73" \
  "$OUT_DIR/p3_hashseed_repeatability.json" "$OUT_DIR/p3_recovery_evidence.json" \
  "$TRAINER_EXIT_CODE" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml

repo, run_dir, replay_dir, hashseed_replay_dir, repeatability_path, output = map(Path, sys.argv[1:7])
trainer_exit_code = int(sys.argv[7])
run_manifest_path = run_dir / "run_manifest.json"
reanalysis_path = replay_dir / "reanalysis_manifest.json"
hashseed_reanalysis_path = hashseed_replay_dir / "reanalysis_manifest.json"
checkpoint = run_dir / "model/ppo_final.zip"
run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
reanalysis = json.loads(reanalysis_path.read_text(encoding="utf-8"))
hashseed_reanalysis = json.loads(hashseed_reanalysis_path.read_text(encoding="utf-8"))
repeatability = json.loads(repeatability_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

required_source_files = {
    "config/features/full_po3_mtf.yaml": (),
    "market_data/loading/bars.py": ("quant_rl/data/loader.py",),
    "market_data/loading/ticks.py": ("quant_rl/data/ticks.py",),
    "simulation/environment/env.py": ("simulation/envs/trading_env.py", "quant_rl/envs/trading_env.py"),
    "features/po3/state.py": ("quant_rl/features/po3_state.py",),
    "evaluation/reanalysis/runner.py": ("quant_rl/eval/eval_run.py",),
    "evaluation/reporting/run_manifest.py": ("quant_rl/eval/run_manifest.py",),
    "config/base/default.yaml": ("quant_rl/config/default.yaml",),
}
saved_source = {
    item["path"]: item.get("sha256")
    for item in run_manifest.get("source", {}).get("dirty_files", [])
}
evaluation_source = {
    item["path"]: item.get("sha256")
    for item in reanalysis.get("evaluation_provenance", {}).get("source", {}).get("dirty_files", [])
}
source_identity = {}
for path, aliases in required_source_files.items():
    source_identity[path] = {
        "saved_sha256": next(
            (saved_source[key] for key in (path, *aliases) if saved_source.get(key)), None
        ),
        "evaluation_sha256": next(
            (evaluation_source[key] for key in (path, *aliases) if evaluation_source.get(key)),
            None,
        ),
        "current_sha256": sha256(repo / path),
    }
source_match = all(
    value["saved_sha256"] is not None
    and value["saved_sha256"] == value["current_sha256"]
    and value["evaluation_sha256"] == value["current_sha256"]
    for value in source_identity.values()
)
checkpoint_match = (
    reanalysis.get("checkpoint_sha256") == sha256(checkpoint)
    and run_manifest.get("checkpoint", {}).get("sha256") == sha256(checkpoint)
)
required_outputs = [
    replay_dir / split / name
    for split in ("training", "testing")
    for name in ("trades.csv", "equity.csv")
]
outputs_complete = all(path.is_file() for path in required_outputs)
data_parity = reanalysis.get("dataset_parity_disposition") == "verified"
source_provenance = reanalysis.get("evaluation_provenance", {}).get("source", {})
evaluation_source_available = source_provenance.get("available") is True
hashseed_provenance = reanalysis.get("evaluation_provenance", {}).get("runtime", {}).get(
    "python_hash_seed_env"
)
second_hashseed_provenance = hashseed_reanalysis.get("evaluation_provenance", {}).get(
    "runtime", {}
).get("python_hash_seed_env")
hashseed_repeatability_passed = (
    repeatability.get("status") == "passed"
    and hashseed_provenance == "0"
    and second_hashseed_provenance == "73"
)
passed = all(
    (
        source_match,
        checkpoint_match,
        outputs_complete,
        data_parity,
        evaluation_source_available,
        hashseed_repeatability_passed,
    )
)
resolved_config_path = run_dir / "config.yaml"
resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
resolved_features = resolved_config.get("features", {})
resolved_htfs = [str(value).upper() for value in resolved_features.get("htf_timeframes", [])]
resolved_ltf = str(resolved_features.get("po3_ltf", "")).upper()
resolved_profile_passed = resolved_ltf == "M1" and resolved_htfs == [
    "M3",
    "M5",
    "M10",
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
]
passed = passed and resolved_profile_passed
report = {
    "schema": "thesis.p3_checkpoint_recovery_smoke.v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "passed" if passed else "failed",
    "run_manifest_path": str(run_manifest_path.relative_to(repo)),
    "run_manifest_sha256": sha256(run_manifest_path),
    "checkpoint_path": str(checkpoint.relative_to(repo)),
    "checkpoint_sha256": sha256(checkpoint),
    "reanalysis_manifest_path": str(reanalysis_path.relative_to(repo)),
    "reanalysis_manifest_sha256": sha256(reanalysis_path),
    "hashseed_reanalysis_manifest_path": str(hashseed_reanalysis_path.relative_to(repo)),
    "hashseed_reanalysis_manifest_sha256": sha256(hashseed_reanalysis_path),
    "hashseed_repeatability_report_path": str(repeatability_path.relative_to(repo)),
    "hashseed_repeatability_report_sha256": sha256(repeatability_path),
    "hashseed_repeatability_status": repeatability.get("status"),
    "hashseed_runtime_values": [hashseed_provenance, second_hashseed_provenance],
    "completed_steps": run_manifest.get("completed_steps"),
    "requested_smoke_steps": 16384,
    "trainer_exit_code": trainer_exit_code,
    "training_survival_gate_disposition": json.loads(
        (run_dir / "training_log.json").read_text(encoding="utf-8")
    ).get("train_equity_reason"),
    "resolved_config_path": str(resolved_config_path.relative_to(repo)),
    "resolved_config_sha256": sha256(resolved_config_path),
    "resolved_po3_ltf": resolved_ltf,
    "resolved_htf_timeframes": resolved_htfs,
    "resolved_mtf_profile_passed": resolved_profile_passed,
    "dataset_parity_disposition": reanalysis.get("dataset_parity_disposition"),
    "train_end": reanalysis.get("train_end"),
    "test_start": reanalysis.get("test_start"),
    "required_source_identity": source_identity,
    "training_and_testing_outputs_complete": outputs_complete,
    "interpretation": "Recovery smoke only; no strategy result or selection claim.",
}
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "status": report["status"],
            "output": str(output),
            "dataset_parity": data_parity,
            "hashseed_repeatability": repeatability.get("status"),
        },
        indent=2,
    )
)
if not passed:
    raise SystemExit("P3 evidence did not meet the recovery gate")
PY

echo "P3 checkpoint recovery smoke passed; allocation remains available for follow-up."
