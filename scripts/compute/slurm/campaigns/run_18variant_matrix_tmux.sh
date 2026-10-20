#!/bin/bash
set -euo pipefail

exec "$(dirname "$0")/run_18variant_matrix_gh200_tmux.sh" "$@"
