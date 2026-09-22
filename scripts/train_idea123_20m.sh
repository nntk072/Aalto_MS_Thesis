#!/bin/bash
# Old path. The queued slot still calls this file. Use train_overlay_baseline_20m.sh.
exec "$(cd "$(dirname "$0")" && pwd)/train_overlay_baseline_20m.sh"
