from __future__ import annotations

from typing import Any

import torch


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "q, k, and v must have shape "
            "(batch_size, sequence_length, d_model)"
        )

    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")

    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same hidden dimension")

    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same sequence length")

    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")

    if not q.is_floating_point():
        raise TypeError("q, k, and v must use a floating-point dtype")


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _validate_attention_inputs(q, k, v)

        if not isinstance(is_causal, bool):
            raise TypeError("is_causal must be a bool")

        raise NotImplementedError(
            "The tiled FlashAttention forward pass is not implemented yet"
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        raise NotImplementedError(
            "The FlashAttention backward pass is not implemented yet"
        )