"""Console logs are mirrored into the run folder.

A run's transcript must live beside its model and eval artifacts, so a single
run directory tells the whole story without cross-referencing a flat
``outputs/*.log`` that interleaves concurrent runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from quant_rl.train.train_rl import _RUN_LOG_NAME, _attach_run_log


@pytest.fixture
def isolated_root() -> Any:
    """Give each test a clean root logger.

    ``_attach_run_log`` adds a handler to the process-wide root logger, and
    pytest's logging plugin forces the root level to WARNING. Both would leak
    between tests: handlers would pile up so each test writes to every earlier
    ``tmp_path``, and INFO records would be dropped. This saves and restores
    handlers and level, and lowers the level so INFO lines actually propagate.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    log = logging.getLogger("test_attach_run_log")
    saved_log_level = log.level
    root.handlers = []
    root.setLevel(logging.INFO)
    log.setLevel(logging.INFO)
    try:
        yield log
    finally:
        for handler in root.handlers:
            handler.flush()
            handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        log.setLevel(saved_log_level)


def test_run_log_is_written_into_run_dir(tmp_path: Path, isolated_root: Any) -> None:
    path = _attach_run_log(tmp_path)
    assert path == tmp_path / _RUN_LOG_NAME
    isolated_root.info("hello from train")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello from train" in path.read_text()


def test_attach_run_log_is_idempotent(tmp_path: Path, isolated_root: Any) -> None:
    """A second call must not add a duplicate handler or truncate the log."""
    before = len(logging.getLogger().handlers)
    first = _attach_run_log(tmp_path)
    assert first is not None
    isolated_root.info("kept")
    for handler in logging.getLogger().handlers:
        handler.flush()
    second = _attach_run_log(tmp_path)
    assert first == second
    assert len(logging.getLogger().handlers) == before + 1
    # Not truncated: the earlier line is still there.
    assert "kept" in first.read_text()


def test_console_handler_is_preserved(tmp_path: Path, isolated_root: Any) -> None:
    """Attaching a file handler must not stop records reaching the console."""
    root = logging.getLogger()
    stream = logging.StreamHandler()
    root.addHandler(stream)
    try:
        before = list(root.handlers)
        _attach_run_log(tmp_path)
        assert all(h in root.handlers for h in before)
        assert len(root.handlers) == len(before) + 1
    finally:
        root.removeHandler(stream)


def test_unwritable_run_dir_does_not_raise(tmp_path: Path, isolated_root: Any) -> None:
    """A read-only output dir must never abort a training run over logging."""
    unwritable = tmp_path / "file_not_a_dir"
    unwritable.write_text("x")
    assert _attach_run_log(unwritable) is None
