from __future__ import annotations

from typing import Any

import torch

Q_TILE_SIZE = 16
K_TILE_SIZE = 16


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

    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[-1] == 0:
        raise ValueError("attention dimensions must be non-empty")


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

        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[1]
        scale = d_model**-0.5

        output = torch.empty_like(q)
        logsumexp = torch.empty(
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )

        for query_start in range(0, n_queries, Q_TILE_SIZE):
            query_end = min(query_start + Q_TILE_SIZE, n_queries)
            query = q[:, query_start:query_end, :].to(torch.float32)
            query_tile_size = query_end - query_start

            output_accumulator = torch.zeros(
                (batch_size, query_tile_size, d_model),
                device=q.device,
                dtype=torch.float32,
            )
            denominator = torch.zeros(
                (batch_size, query_tile_size),
                device=q.device,
                dtype=torch.float32,
            )
            row_maximum = torch.full(
                (batch_size, query_tile_size),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )

            for key_start in range(0, n_keys, K_TILE_SIZE):
                key_end = min(key_start + K_TILE_SIZE, n_keys)
                key = k[:, key_start:key_end, :].to(torch.float32)
                value = v[:, key_start:key_end, :].to(torch.float32)

                scores = torch.matmul(query, key.transpose(-2, -1)) * scale
                new_row_maximum = torch.maximum(
                    row_maximum,
                    scores.amax(dim=-1),
                )
                correction = torch.exp(row_maximum - new_row_maximum)
                probabilities = torch.exp(
                    scores - new_row_maximum.unsqueeze(-1)
                )

                denominator = (
                    correction * denominator
                    + probabilities.sum(dim=-1)
                )
                output_accumulator = (
                    correction.unsqueeze(-1) * output_accumulator
                    + torch.matmul(probabilities, value)
                )
                row_maximum = new_row_maximum

            output[:, query_start:query_end, :] = (
                output_accumulator / denominator.unsqueeze(-1)
            ).to(q.dtype)
            logsumexp[:, query_start:query_end] = (
                row_maximum + torch.log(denominator)
            )

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        raise NotImplementedError(
            "The FlashAttention backward pass is not implemented yet"
        )
