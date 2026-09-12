#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/bin:$HOME/.local/bin:$PATH"

REPO="$HOME/Aalto_MS_Thesis"

declare -A AGENTS=(
    [planner-opencode]="opencode"
    [planner-kilo]="kilo"
    [planner-cline]="cline"
    [planner-gemini]="gemini"
    [planner-vibe]="vibe"
)

echo "========================================"
echo "      Oasis Multi-Agent Setup"
echo "========================================"
echo

# --------------------------------------------------
# 1. Check environment
# --------------------------------------------------

echo "[1/4] Checking environment..."

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed."
    echo "Install with: sudo apt install -y tmux"
    exit 1
fi

if [ ! -d "$REPO" ]; then
    echo "ERROR: Repository not found:"
    echo "  $REPO"
    exit 1
fi

echo "  tmux : $(tmux -V)"
echo "  repo : $REPO"

# --------------------------------------------------
# 2. Check CLI agents
# --------------------------------------------------

echo
echo "[2/4] Checking CLI agents..."

for session in "${!AGENTS[@]}"; do
    cli="${AGENTS[$session]}"

    if command -v "$cli" >/dev/null 2>&1; then
        echo "  $cli : $(command -v "$cli")"
    else
        echo "  WARNING: $cli not found"
    fi
done

# --------------------------------------------------
# 3. Create tmux sessions and launch agents
# --------------------------------------------------

echo
echo "[3/4] Creating tmux sessions..."

for session in "${!AGENTS[@]}"; do
    cli="${AGENTS[$session]}"

    # Skip missing CLI
    if ! command -v "$cli" >/dev/null 2>&1; then
        echo "  SKIP $session ($cli not installed)"
        continue
    fi

    # Create session if missing
    if ! tmux has-session -t "$session" 2>/dev/null; then
        tmux new-session \
            -d \
            -s "$session" \
            -c "$REPO"

        echo "  CREATED: $session"
    else
        echo "  EXISTS : $session"
    fi

    # Determine current foreground command
    current_cmd="$(
        tmux list-panes \
            -t "$session" \
            -F '#{pane_current_command}' \
            2>/dev/null || true
    )"

    # Start CLI only when it is not already running
    if [[ "$current_cmd" != "$cli" ]]; then
        tmux send-keys \
            -t "$session:0" \
            "$cli" \
            C-m

        echo "  STARTED: $cli"
    else
        echo "  RUNNING: $cli"
    fi
done

# --------------------------------------------------
# 4. Show final state
# --------------------------------------------------

echo
echo "[4/4] Final state"
echo

tmux ls 2>/dev/null || true

echo
echo "========================================"
echo "READY"
echo "========================================"
echo
echo "Repo:"
echo "  $REPO"
echo
echo "Oasis sessions:"
for session in "${!AGENTS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "  ✓ $session"
    else
        echo "  ✗ $session"
    fi
done

# --------------------------------------------------
# 5. Launch Oasis GUI
# --------------------------------------------------

echo
echo "[5/5] Launching Oasis GUI..."

RUNNER="/usr/lib/OasisGUI/binaries/oasis-runner-x86_64-unknown-linux-musl"
SOCKET_DIR="/run/user/$(id -u)"
SOCKET="$SOCKET_DIR/oasis-runner.sock"

mkdir -p "$SOCKET_DIR"

if pgrep -f 'oasis-runner.*serve' >/dev/null 2>&1; then
    echo "  RUNNING: Oasis GUI server already active"
    echo "  SOCKET : $SOCKET"
elif [ -x "$RUNNER" ]; then
    nohup "$RUNNER" serve --socket "$SOCKET" >/tmp/oasis-runner.log 2>&1 &
    echo "  LAUNCHED: Oasis GUI server (pid $!)"
    echo "  SOCKET : $SOCKET"
    sleep 3
    if command -v oasis-gui >/dev/null 2>&1; then
        timeout 5 oasis-gui focus --server planner-cline >/dev/null 2>&1 || true
    fi
else
    echo "  WARNING: oasis-runner not found at $RUNNER"
    echo "  Start the Oasis GUI manually."
fi

echo
echo "Open Oasis GUI to manage the sessions."
