"""Tests for device resolution and RAM/VRAM-aware training throughput."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from quant_rl.utils.device import (
    _GiB,
    configure_cuda,
    get_device,
    scale_training_cfg,
    suggest_batch_size,
    suggest_n_envs,
    suggest_n_epochs,
    suggest_n_steps,
)


class TestGetDevice:
    """Tests for get_device()."""

    def test_explicit_cpu(self) -> None:
        assert get_device("cpu").type == "cpu"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid device"):
            get_device("not-a-device")


class TestSuggestNEnvs:
    """RAM capping must beat the GPU target on a 16 GB laptop."""

    def test_debug_stays_one(self) -> None:
        assert (
            suggest_n_envs(
                requested=1,
                total_ram_bytes=128 * _GiB,
                available_ram_bytes=100 * _GiB,
                cpu_count=64,
                vram_bytes=80 * _GiB,
            )
            == 1
        )

    def test_caps_when_ram_is_tight(self) -> None:
        n = suggest_n_envs(
            requested=8,
            total_ram_bytes=16 * _GiB,
            available_ram_bytes=4 * _GiB,
            cpu_count=8,
            vram_bytes=6 * _GiB,
        )
        assert n <= 2

    def test_16gb_laptop_stays_at_one_when_avail_is_typical(self) -> None:
        n = suggest_n_envs(
            requested=8,
            total_ram_bytes=16 * _GiB,
            available_ram_bytes=9 * _GiB,
            cpu_count=8,
            vram_bytes=6 * _GiB,
        )
        # 4 GiB/worker planning: 9G avail → 1 worker after parent/headroom.
        assert n == 1

    def test_a100_scales_up_when_ram_allows(self) -> None:
        n = suggest_n_envs(
            requested=4,
            total_ram_bytes=256 * _GiB,
            available_ram_bytes=200 * _GiB,
            cpu_count=32,
            vram_bytes=40 * _GiB,
        )
        assert n >= 8
        assert n <= 16
        assert n & (n - 1) == 0

    def test_gh200_caps_at_32_on_256g_budget(self) -> None:
        n = suggest_n_envs(
            requested=8,
            total_ram_bytes=256 * _GiB,
            available_ram_bytes=256 * _GiB,
            cpu_count=48,
            vram_bytes=142 * _GiB,
        )
        assert n == 32

    def test_gh200_allows_64_for_512g_allocation(self) -> None:
        n = suggest_n_envs(
            requested=8,
            total_ram_bytes=512 * _GiB,
            available_ram_bytes=512 * _GiB,
            cpu_count=48,
            vram_bytes=142 * _GiB,
        )
        assert n == 64

    def test_cpu_does_not_scale_up(self) -> None:
        n = suggest_n_envs(
            requested=4,
            total_ram_bytes=256 * _GiB,
            available_ram_bytes=200 * _GiB,
            cpu_count=32,
            vram_bytes=None,
        )
        assert n == 4


class TestSuggestBatchSize:
    """Minibatch must divide the PPO rollout buffer."""

    def test_divides_rollout_on_laptop(self) -> None:
        bs = suggest_batch_size(
            current=256,
            n_steps=2048,
            n_envs=8,
            vram_bytes=6 * _GiB,
            arch="transformer",
        )
        rollout = (2048 // 8) * 8
        assert rollout % bs == 0
        assert bs >= 256

    def test_a100_uses_full_rollout(self) -> None:
        bs = suggest_batch_size(
            current=256,
            n_steps=8192,
            n_envs=16,
            vram_bytes=40 * _GiB,
            arch="transformer",
        )
        assert bs == 4096
        assert 8192 % bs == 0

    def test_cpu_keeps_current(self) -> None:
        assert (
            suggest_batch_size(
                current=256,
                n_steps=2048,
                n_envs=4,
                vram_bytes=None,
                arch="transformer",
            )
            == 256
        )


class TestSuggestNStepsEpochs:
    """A100 grows the rollout; CUDA adds PPO epochs."""

    def test_n_steps_a100(self) -> None:
        assert suggest_n_steps(current=2048, vram_bytes=40 * _GiB) == 8192

    def test_n_steps_laptop_unchanged(self) -> None:
        assert suggest_n_steps(current=2048, vram_bytes=6 * _GiB) == 2048

    def test_n_epochs_cuda(self) -> None:
        assert suggest_n_epochs(current=10, has_cuda=True) == 15

    def test_n_epochs_cpu(self) -> None:
        assert suggest_n_epochs(current=10, has_cuda=False) == 10


def test_configure_cuda_is_noop_safe() -> None:
    """Must not raise on CPU-only hosts."""
    configure_cuda()


def test_scale_training_cfg_cpu_keeps_n_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU path must not inflate n_envs when RAM is plentiful."""
    import torch

    from quant_rl.utils import device as device_mod

    monkeypatch.setattr(device_mod, "_read_meminfo", lambda: (256 * _GiB, 200 * _GiB))
    cfg = OmegaConf.create(
        {
            "env": {"n_envs": 4},
            "ppo": {"n_steps": 2048, "batch_size": 256, "n_epochs": 10},
            "sac": {"batch_size": 256},
        }
    )
    scale_training_cfg(cfg, torch.device("cpu"), "transformer")
    assert int(cfg.env.n_envs) == 4
    assert int(cfg.ppo.batch_size) == 256
    assert int(cfg.ppo.n_epochs) == 10


def test_enable_extractor_autocast_skips_transformer() -> None:
    """Transformer stays fp32; bf16 autocast caused PPO policy NaNs on GH200."""
    pytest.importorskip("torch")
    from gymnasium import spaces

    from quant_rl.models.encoder import TransformerEncoder
    from quant_rl.utils.device import enable_extractor_autocast

    observation_space = spaces.Dict(
        {
            "seq": spaces.Box(low=-1.0, high=1.0, shape=(8, 4), dtype=np.float32),
            "account": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        }
    )
    encoder = TransformerEncoder(
        observation_space=observation_space,
        seq_len=8,
        n_features=4,
        latent_dim=16,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
    )
    enable_extractor_autocast(encoder)
    assert getattr(encoder, "_amp_wrapped", False) is False
