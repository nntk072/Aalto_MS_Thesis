"""Tests for device resolution and RAM/VRAM-aware training throughput."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from omegaconf import OmegaConf

from compute.environment.device import (
    _GiB,
    get_device,
    scale_training_cfg,
    suggest_batch_size,
    suggest_n_envs,
    suggest_n_epochs,
    suggest_n_steps,
)
from compute.environment.precision import (
    configure_cuda_for_training,
    get_autocast_context,
    get_autocast_dtype,
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

    def test_gh200_allows_128_for_600g_allocation(self) -> None:
        n = suggest_n_envs(
            requested=8,
            total_ram_bytes=600 * _GiB,
            available_ram_bytes=600 * _GiB,
            cpu_count=48,
            vram_bytes=142 * _GiB,
        )
        assert n == 128

    def test_gh200_stays_at_64_on_512g_even_when_requested_128(self) -> None:
        n = suggest_n_envs(
            requested=128,
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

    def test_bundle_worker_budget_uses_measured_shared_array_cost(self) -> None:
        n = suggest_n_envs(
            requested=64,
            total_ram_bytes=16 * _GiB,
            available_ram_bytes=10 * _GiB,
            cpu_count=16,
            vram_bytes=142 * _GiB,
            worker_ram_bytes=128 * 1024**2,
        )
        assert n == 32


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


def test_configure_cuda_for_training_is_noop_safe() -> None:
    """Must not raise on CPU-only hosts."""
    configure_cuda_for_training()


def test_autocast_is_a_noop_on_cpu() -> None:
    import torch

    assert get_autocast_dtype("cpu") is None
    with get_autocast_context("cpu"):
        assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize(
    ("capability", "supports_bfloat16", "expected"),
    [
        ((7, 0), False, "torch.float16"),  # V100
        ((8, 0), True, "torch.bfloat16"),  # A100
        ((9, 0), True, "torch.bfloat16"),  # H100/H200/GH200
        ((10, 0), True, "torch.bfloat16"),  # B300 and newer
        ((9, 0), False, "torch.float16"),  # Runtime fallback
    ],
)
def test_autocast_dtype_tracks_device_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    supports_bfloat16: bool,
    expected: str,
) -> None:
    import torch

    from compute.contracts import DeviceCapabilities
    from compute.environment import precision

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        precision,
        "discover_device_capabilities",
        lambda _device: DeviceCapabilities(
            architecture="x86_64",
            accelerator_kind="cuda",
            accelerator_count=1,
            accelerator_memory_bytes=None,
            supports_bfloat16=supports_bfloat16,
            compute_capability=capability,
        ),
    )
    assert str(get_autocast_dtype("cuda:0")) == expected


def test_configure_cuda_enables_nvidia_tf32_and_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import torch

    from compute.contracts import DeviceCapabilities
    from compute.environment import precision

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cuda_matmul = SimpleNamespace(allow_tf32=False)
    cudnn = SimpleNamespace(is_available=lambda: True, allow_tf32=False, benchmark=False)
    monkeypatch.setattr(torch.backends.cuda, "matmul", cuda_matmul)
    monkeypatch.setattr(torch.backends, "cudnn", cudnn)
    monkeypatch.setattr(
        precision,
        "discover_device_capabilities",
        lambda _device=None: DeviceCapabilities(
            architecture="x86_64",
            accelerator_kind="cuda",
            accelerator_count=1,
            accelerator_memory_bytes=None,
            supports_bfloat16=True,
            accelerator_backend="cuda",
            compute_capability=(8, 0),
        ),
    )
    configure_cuda_for_training()
    assert __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cuda_matmul.allow_tf32 is True
    assert cudnn.allow_tf32 is True
    assert cudnn.benchmark is True


def test_configure_rocm_skips_cuda_allocator_and_tf32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from compute.contracts import DeviceCapabilities
    from compute.environment import precision

    monkeypatch.setattr(torch.version, "hip", "6.3")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        precision,
        "discover_device_capabilities",
        lambda _device=None: DeviceCapabilities(
            architecture="x86_64",
            accelerator_kind="rocm",
            accelerator_count=1,
            accelerator_memory_bytes=None,
            supports_bfloat16=True,
            accelerator_backend="rocm",
            device_name="AMD Instinct MI210",
        ),
    )
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    configure_cuda_for_training()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in __import__("os").environ
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_scale_training_cfg_cpu_keeps_n_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU path must not inflate n_envs when RAM is plentiful."""
    import torch

    from compute.environment import device as device_mod

    monkeypatch.setattr(device_mod, "_read_meminfo", lambda: (256 * _GiB, 200 * _GiB))
    monkeypatch.setattr(device_mod, "_read_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(device_mod, "_read_slurm_memory_limit", lambda: None)
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


def test_scale_training_cfg_preserves_exact_quality_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from compute.environment import device as device_mod

    monkeypatch.setattr(device_mod, "_read_meminfo", lambda: (256 * _GiB, 200 * _GiB))
    monkeypatch.setattr(device_mod, "_read_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(device_mod, "_read_slurm_memory_limit", lambda: None)
    cfg = OmegaConf.create(
        {
            "env": {"n_envs": 16, "data_source": "bundle"},
            "ppo": {"n_steps": 512, "batch_size": 512, "n_epochs": 10},
        }
    )
    scale_training_cfg(cfg, torch.device("cpu"), "tcn", preserve_ppo_geometry=True)
    assert int(cfg.env.n_envs) == 16
    assert int(cfg.ppo.n_steps) == 512
    assert int(cfg.ppo.batch_size) == 512
    assert int(cfg.ppo.n_epochs) == 10


def test_scale_training_cfg_exact_geometry_rejects_explicit_env_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from compute.environment import device as device_mod

    monkeypatch.setattr(device_mod, "_read_meminfo", lambda: (256 * _GiB, 200 * _GiB))
    monkeypatch.setenv("QUANT_RL_MAX_N_ENVS", "8")
    cfg = OmegaConf.create(
        {
            "env": {"n_envs": 16, "data_source": "bundle"},
            "ppo": {"n_steps": 512, "batch_size": 512, "n_epochs": 10},
        }
    )
    with pytest.raises(ValueError, match="exact quality-experiment PPO geometry"):
        scale_training_cfg(cfg, torch.device("cpu"), "tcn", preserve_ppo_geometry=True)


def test_scale_training_cfg_uses_bundle_worker_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundle mode passes the measured per-worker allowance into RAM scaling."""
    import torch

    from compute.environment import device as device_mod

    observed: dict[str, int | None] = {}
    original = device_mod.suggest_n_envs

    def capture_worker_budget(**kwargs: int | None) -> int:
        observed["worker_ram_bytes"] = kwargs.get("worker_ram_bytes")
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(device_mod, "suggest_n_envs", capture_worker_budget)
    monkeypatch.setattr(device_mod, "_read_meminfo", lambda: (256 * _GiB, 200 * _GiB))
    cfg = OmegaConf.create(
        {
            "env": {"n_envs": 4, "data_source": "bundle"},
            "ppo": {"n_steps": 2048, "batch_size": 256, "n_epochs": 10},
        }
    )
    scale_training_cfg(cfg, torch.device("cpu"), "tcn")
    assert observed["worker_ram_bytes"] == device_mod._BUNDLE_WORKER_RAM_BYTES


def test_enable_extractor_autocast_skips_transformer() -> None:
    """Transformer stays fp32; bf16 autocast caused PPO policy NaNs on GH200."""
    torch = pytest.importorskip("torch")
    from gymnasium import spaces

    from models.encoders import TransformerEncoder
    from models.runtime.precision import enable_extractor_autocast

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
    enable_extractor_autocast(encoder, torch.bfloat16)
    assert getattr(encoder, "_amp_wrapped", False) is False


def test_fp16_extractor_autocast_is_disabled_for_grad_enabled_ppo_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib

    import torch

    from models.runtime import precision as model_precision

    contexts: list[tuple[str, torch.dtype]] = []

    def autocast_context(
        *, device_type: str, dtype: torch.dtype
    ) -> contextlib.AbstractContextManager[object]:
        contexts.append((device_type, dtype))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch, "autocast", autocast_context)
    extractor = torch.nn.Linear(3, 2)
    model_precision.enable_extractor_autocast(extractor, torch.float16)

    inputs = torch.ones((4, 3), requires_grad=True)
    train_output = extractor(inputs)
    train_output.sum().backward()
    assert train_output.dtype == torch.float32
    assert train_output.grad_fn is not None
    assert contexts == []

    with torch.no_grad():
        inference_output = extractor(inputs)
    assert inference_output.dtype == torch.float32
    assert len(contexts) == 1
    assert contexts[0] == ("cuda", torch.float16)


def test_fp16_inference_retries_in_fp32_if_features_are_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib

    import torch

    from models.runtime import precision as model_precision

    amp_enabled = False

    @contextlib.contextmanager
    def autocast_context(*, device_type: str, dtype: torch.dtype) -> Iterator[None]:
        nonlocal amp_enabled
        del device_type, dtype
        amp_enabled = True
        try:
            yield
        finally:
            amp_enabled = False

    class OverflowOnAutocast(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if amp_enabled:
                return value * torch.tensor(float("inf"))
            return value * self.weight

    monkeypatch.setattr(torch, "autocast", autocast_context)
    extractor = OverflowOnAutocast()
    model_precision.enable_extractor_autocast(extractor, torch.float16)

    with torch.no_grad():
        output = extractor(torch.ones((2, 1)))

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert extractor.calls == 2
