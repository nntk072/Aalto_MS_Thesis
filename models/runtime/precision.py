"""Model-specific mixed precision adaptation without hardware discovery."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.encoders.transformer import TransformerEncoder


def enable_extractor_autocast(extractor: nn.Module, dtype: torch.dtype | None) -> None:
    """Apply a caller-selected autocast dtype to an encoder; return fp32 features."""
    # Transformer attention in bf16 can NaN under PPO's large rollout batches.
    if isinstance(extractor, TransformerEncoder):
        return
    if getattr(extractor, "_amp_wrapped", False):
        return
    if dtype is None:
        return
    original_forward = extractor.forward

    def forward_amp(*args: Any, **kwargs: Any) -> torch.Tensor:
        # SB3's PPO optimizer calls backward/step without GradScaler. Running
        # V100 fp16 encoder ops with autograd can therefore overflow gradients
        # and corrupt actor logits. Keep PPO updates in fp32 until its optimizer
        # path owns a GradScaler; retain fp16 for no-grad rollout/inference.
        if dtype == torch.float16 and torch.is_grad_enabled():
            output = original_forward(*args, **kwargs)
            if not isinstance(output, torch.Tensor):
                raise TypeError("autocast feature extractors must return a torch.Tensor")
            return output.float()
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = original_forward(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("autocast feature extractors must return a torch.Tensor")
        if dtype == torch.float16 and not torch.isfinite(output).all().item():
            # V100 fp16 has a narrow exponent range. If a trained encoder
            # overflows during no-grad rollout inference, retry that pass in
            # fp32 so one bad autocast result cannot poison the policy logits.
            output = original_forward(*args, **kwargs)
            if not isinstance(output, torch.Tensor):
                raise TypeError("autocast feature extractors must return a torch.Tensor")
            if not torch.isfinite(output).all().item():
                raise FloatingPointError(
                    "feature extractor returned non-finite features in fp16 autocast and fp32"
                )
        return output.float()

    setattr(extractor, "forward", forward_amp)
    setattr(extractor, "_amp_wrapped", True)


__all__ = [
    "enable_extractor_autocast",
]
