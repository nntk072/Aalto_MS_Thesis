#!/usr/bin/env bash
# Preserve a campaign's original Git SHA during resume fingerprint creation.
# The controller still hashes every training/config input, so changed campaign
# sources continue to fail the existing fingerprint check.

campaign_manifest="${TRITON_REPO:?}/outputs/gpu_campaigns/${CAMPAIGN_ID:?}/campaign.json"
if [[ ! -s "$campaign_manifest" ]]; then
  return 0
fi

# activate_venv.sh may be sourced again by nested validation scripts. Keep the
# first real Git path and shim environment instead of wrapping the shim itself.
if [[ "${CAMPAIGN_GIT_IDENTITY_ACTIVE:-}" == "1" ]]; then
  return 0
fi

campaign_recorded_git_sha="$(python - "$campaign_manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
print(manifest.get("git_sha", ""))
PY
)"
if [[ ! "$campaign_recorded_git_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: campaign manifest has no valid recorded git_sha: $campaign_manifest" >&2
  return 1
fi

campaign_real_git="$(command -v git)"
campaign_git_shim_dir="${TMPDIR:-/tmp}/campaign-git-identity-${SLURM_JOB_ID:-$$}"
mkdir -p "$campaign_git_shim_dir"
cat >"$campaign_git_shim_dir/git" <<'SH'
#!/usr/bin/env bash
if [[ $# -eq 2 && "$1" == "rev-parse" && "$2" == "HEAD" ]]; then
  parent_exe="$(readlink "/proc/$PPID/exe" 2>/dev/null || true)"
  parent_cmd="$(tr '\0' ' ' <"/proc/$PPID/cmdline" 2>/dev/null || true)"
  if [[ "$parent_exe" == */python* \
    && "$(pwd -P)" == "${CAMPAIGN_GIT_IDENTITY_REPO:-}" \
    && "$parent_cmd" == *campaign_controller.py* ]]; then
    printf '%s\n' "${CAMPAIGN_GIT_IDENTITY_SHA:?}"
    exit 0
  fi
fi
exec "${CAMPAIGN_GIT_IDENTITY_REAL_GIT:?}" "$@"
SH
chmod 700 "$campaign_git_shim_dir/git"

export CAMPAIGN_GIT_IDENTITY_SHA="$campaign_recorded_git_sha"
export CAMPAIGN_GIT_IDENTITY_REPO="$(cd "$TRITON_REPO" && pwd -P)"
export CAMPAIGN_GIT_IDENTITY_REAL_GIT="$campaign_real_git"
export CAMPAIGN_GIT_IDENTITY_ACTIVE=1
export PATH="$campaign_git_shim_dir:$PATH"
echo "Campaign resume will retain recorded Git identity $campaign_recorded_git_sha; source hashes remain enforced."
