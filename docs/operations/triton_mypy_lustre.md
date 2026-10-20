# Mypy on Triton: Lustre startup stalls (ARM64 and x86_64)

## Verified incident (2026-10-11)

On `gpuarm1` (aarch64/GH200), the installed `.venv/bin/mypy --version`
sometimes stalled for minutes before printing anything. The process was observed
in `folio_wait_bit_common` or `cl_sync_io_wait` while loading modules from
Lustre scratch. A local-staged Python 3.12.15 and mypy 2.3.0 started immediately,
and full-tree `mypy .` passed: **355 source files, zero type errors**.
This diagnoses an intermittent filesystem I/O startup problem, not a mypy
type-checking failure. Ordinary direct invocation has **not** been repaired.

Do not interpret a blank terminal or `mypy --version` delay as a type error.
Do not assume every delay is caused by Lustre without inspecting the process.
The x86_64 staging procedure below is designed to be architecture-neutral
but **has not been validated on an x86 compute node**.

## Choose the correct environment

Run on the allocated compute node, not based on the SSH login node's architecture:

| `uname -m` | Node example | Virtual environment |
| --- | --- | --- |
| `aarch64` | GH200 / gpuarm | `.venv` |
| `x86_64` | H100/H200 / gpu | `.venv-x86` |

`source scripts/compute/slurm/activate_venv.sh` selects the matching environment.
Avoid sourcing `.venv` on the x86_64 login node. Never share local staging
files between different architectures, users, or compute nodes.

## Diagnose without interrupting a training session

From the *existing* allocated compute shell:

```bash
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
uname -m
source scripts/compute/slurm/activate_venv.sh
command -v mypy
timeout 30 mypy --version
echo "version_exit=$?"  # 124 means timeout
ps -u "$USER" -o pid,stat,wchan:24,etime,pcpu,args
df -h /tmp /scratch
```

Inspect the process state and Lustre latency before concluding a cause.
Use an existing tmux/Slurm shell; follow `.agents/rules/triton-slurm.md` before
starting an additional job or Slurm step.

## Portable node-local mypy runner

The following creates *temporary* files under a private node-local directory.
It stages the Python runtime and mypy code on `/tmp`, while using installed
dependencies in the matching persistent virtual environment. It avoids
changing the project's lockfile, source, or production settings.

```bash
cd /scratch/work/nguyenl37/Aalto_MS_Thesis
source scripts/compute/slurm/activate_venv.sh
ENV_DIR="$(dirname "$(dirname "$(command -v python)")")"
STAGE="$(mktemp -d "/tmp/triton-mypy-${USER}-XXXXXX")"
mkdir -p "$STAGE/sitepkgs" "$STAGE/cache"
BASE="$(dirname "$(dirname "$(readlink -f "$ENV_DIR/bin/python")")")"
cp -a "$BASE" "$STAGE/pythonbase"
SITE="$ENV_DIR/lib/python3.12/site-packages"
# Copy mypy's compiled module and minimal runtime dependencies.
cp -a "$SITE/mypy" "$SITE/mypyc" "$SITE/librt" "$SITE/pathspec" \
  "$SITE/packaging" "$SITE/mypy_extensions.py" \
  "$SITE/typing_extensions.py" "$STAGE/sitepkgs/"
find "$SITE" -maxdepth 1 -name '*__mypyc*.so' -exec cp -a {} "$STAGE/sitepkgs/" \;
PYTHONPATH="$STAGE/sitepkgs:$SITE" \
  "$STAGE/pythonbase/bin/python3.12" -m mypy --version
PYTHONPATH="$STAGE/sitepkgs:$SITE" \
  "$STAGE/pythonbase/bin/python3.12" -m mypy . \
  --cache-dir "$STAGE/cache" --no-color-output --show-error-codes
# After the process has finished, remove only this unique temporary directory:
rm -rf -- "$STAGE"
```

The example expects Python 3.12 and mypy's compiled wheel layout as installed
on 2026-10-11. Verify paths after package upgrades. If importing mypy fails
because another runtime module is missing, stage that specific package from
the matching virtual environment. Run the full project mypy gate (`mypy .`);
never replace it with a scoped run to claim CI parity.

Do not run `rm -rf /tmp/*`, replace the original venv, change Python versions,
or cancel jobs to fix this I/O problem. The node-local directory is ephemeral
and must be recreated after a compute node or allocation changes.

## Verification and reporting

Record architecture, node, virtual environment, Python/mypy versions, command,
exit code, checked-file count, and errors. A successful `--version` only proves
startup; a successful full-tree run proves the configured type check. The
2026-10-11 aarch64 full-tree result was `Success: no issues found in 355 source
files`. The analogous x86_64 full-tree check remains to be performed.
