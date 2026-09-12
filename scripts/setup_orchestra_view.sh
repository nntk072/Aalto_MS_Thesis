#!/usr/bin/env bash
# Orchestra visible-run companion — like setup_oasis_agents.sh, but for a live task.
#
# Shows every orchestra agent tmux session for one task with readable names:
#   orchestra-{role}-{provider}-{model}-{cli}-{task_id}-{ts}
# e.g. orchestra-planner-opencode-default-opencode-task-123-456
#
# Usage:
#   ./scripts/setup_orchestra_view.sh <task-id> [--gui]
#
# With --gui, also starts the Oasis runner + GUI (same as setup_oasis_agents.sh step 5).
set -euo pipefail

export PATH="/usr/bin:$HOME/.local/bin:$PATH"
REPO="$HOME/Aalto_MS_Thesis"
STATE_DIR="$REPO/orchestra/state"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed. Install with: sudo apt install -y tmux"
    exit 1
fi

TASK_ID="${1:-}"
GUI=0
if [ "${2:-}" = "--gui" ]; then GUI=1; fi

if [ -z "$TASK_ID" ]; then
    echo "Usage: $0 <task-id> [--gui]"
    echo
    echo "Recent tasks:"
    ls -t "$STATE_DIR"/task-*.json 2>/dev/null | head -5 | xargs -n1 basename 2>/dev/null || echo "  (no tasks yet)"
    exit 1
fi

STATE_FILE="$STATE_DIR/${TASK_ID}.json"
if [ ! -f "$STATE_FILE" ]; then
    echo "ERROR: no state file $STATE_FILE"
    echo "Run first: orchestra run \"<task>\" --keep-alive   (prints Task ID)"
    exit 1
fi

echo "========================================"
echo "  Orchestra live view — $TASK_ID"
echo "========================================"
echo

# Sessions for this task (readable names from sessions.display_session_name)
mapfile -t SESSIONS < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -F "$TASK_ID" || true)

if [ "${#SESSIONS[@]}" -eq 0 ]; then
    echo "No live tmux sessions for $TASK_ID (agents may have finished)."
    echo "State file still has all outputs:"
    echo "  $STATE_FILE"
else
    printf '%-60s %-12s %s\n' "SESSION (role-provider-model-cli)" "ROLE" "ATTACH"
    echo "----------------------------------------------------------------------------------------------------"
    for s in "${SESSIONS[@]}"; do
        # orchestra-{role}-{provider}-{model}-{cli}-{task}-{ts}
        role="$(echo "$s" | cut -d- -f2)"
        printf '%-60s %-12s %s\n' "$s" "$role" "tmux attach -t $s"
    done
    echo
    echo "All sessions:  tmux ls | grep $TASK_ID"
    echo "Watch logs:    tmux capture-pane -t <session> -p -S -50"
fi

echo
echo "Phase / outputs from state file:"
python3 - "$STATE_FILE" <<'EOF'
import json, sys
d = json.loads(open(sys.argv[1]).read())
print("  phase:", d.get("phase"))
print("  planners:", len(d.get("planner_outputs", [])))
print("  critics:", len(d.get("critic_outputs", [])))
print("  reviewers:", len(d.get("review_outputs", [])))
print("  verdict:", d.get("review_verdict"))
print("  fix_loops:", d.get("fix_loop_count"))
EOF

if [ "$GUI" = "1" ]; then
    echo
    echo "[GUI] Launching Oasis GUI..."
    RUNNER="/usr/lib/OasisGUI/binaries/oasis-runner-x86_64-unknown-linux-musl"
    SOCKET_DIR="/run/user/$(id -u)"
    SOCKET="$SOCKET_DIR/oasis-runner.sock"
    mkdir -p "$SOCKET_DIR"
    if pgrep -f 'oasis-runner.*serve' >/dev/null 2>&1; then
        echo "  RUNNING: Oasis GUI server already active ($SOCKET)"
    elif [ -x "$RUNNER" ]; then
        nohup "$RUNNER" serve --socket "$SOCKET" >/tmp/oasis-runner.log 2>&1 &
        echo "  LAUNCHED: Oasis GUI server (pid $!, socket $SOCKET)"
        echo "  Open Oasis GUI to manage the sessions above."
    else
        echo "  WARNING: oasis-runner not found at $RUNNER"
    fi
else
    echo
    echo "Tip: re-run with --gui to also open the Oasis GUI."
fi
