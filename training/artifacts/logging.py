"""Training run log setup."""

from __future__ import annotations

import logging
from pathlib import Path

_RUN_LOG_NAME = "train.log"


def _attach_run_log(run_dir: Path) -> Path | None:
    """Mirror console logs into ``run_dir/train.log``.

    Adds a file handler to the root logger without removing console handlers,
    so the transcript lives beside the run's model and eval artifacts. A
    second call for the same directory reuses the existing handler instead of
    duplicating records or truncating the file. An unwritable directory
    returns ``None`` rather than aborting training over logging.
    """
    name = str(run_dir / _RUN_LOG_NAME)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == name:
            return Path(name)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(name)
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return Path(name)
