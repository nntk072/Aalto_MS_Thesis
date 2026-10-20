"""Checkpoint discovery for historical model reanalysis."""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    """Return the model checkpoint to evaluate.

    Prefers an explicit ``--checkpoint``, then ``model/ppo_final.zip``
    (written on successful training completion), then the highest-step
    ``ppo_ckpt_*_steps.zip`` (for a run whose training hasn't finished, or
    was interrupted, but has intermediate checkpoints).
    """
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / checkpoint
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        return path

    model_dir = run_dir / "model"
    final = model_dir / "ppo_final.zip"
    if final.exists():
        return final

    ckpts = list(model_dir.glob("ppo_ckpt_*_steps.zip"))
    if not ckpts:
        raise SystemExit(f"No model checkpoints found under {model_dir}")

    def _steps(p: Path) -> int:
        m = _CKPT_STEPS_RE.search(p.name)
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=_steps)
    log.warning("No ppo_final.zip in %s; using latest checkpoint: %s", model_dir, latest.name)
    return latest
