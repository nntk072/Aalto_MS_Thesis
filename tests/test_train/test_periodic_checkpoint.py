"""Periodic env-step checkpoints for long PPO runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from quant_rl.train.callbacks import PeriodicCheckpointCallback
from quant_rl.train.train_rl import _periodic_checkpoint_callback


class _FakeModel:
    def __init__(self) -> None:
        self.num_timesteps = 0
        self.saved: list[str] = []

    def save(self, path: Any) -> None:
        self.saved.append(Path(path).name)


def test_periodic_checkpoint_uses_env_timesteps_not_vec_calls(tmp_path: Path) -> None:
    cb = PeriodicCheckpointCallback(save_freq=1000, save_path=tmp_path, verbose=0)
    model = _FakeModel()
    cb.init_callback(model)  # type: ignore[arg-type]
    for t in range(64, 2500, 64):
        model.num_timesteps = t
        cb.on_step()
    numbered = [n for n in model.saved if n.startswith("ppo_ckpt_")]
    assert numbered[0] == "ppo_ckpt_1024_steps"
    assert "ppo_latest" in model.saved
    assert model.saved.count("ppo_ckpt_1024_steps") == 1
    assert any(n.startswith("ppo_ckpt_2048") for n in numbered)


def test_periodic_checkpoint_disabled_when_freq_zero(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"ppo": {"checkpoint_freq": 0}})
    assert _periodic_checkpoint_callback(cfg, tmp_path) is None


def test_periodic_checkpoint_callback_reads_default_million(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"ppo": {"checkpoint_freq": 1_000_000}})
    cb = _periodic_checkpoint_callback(cfg, tmp_path)
    assert cb is not None
    assert cb.save_freq == 1_000_000
