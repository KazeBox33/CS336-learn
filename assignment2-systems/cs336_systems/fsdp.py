"""Educational fully-sharded data-parallel building blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorShardMetadata:
    """Describe how one logical tensor is flattened and divided across ranks."""

    original_shape: torch.Size
    original_numel: int
    shard_numel: int
    world_size: int

    @property
    def padded_numel(self) -> int:
        return self.shard_numel * self.world_size


def _shard_tensor(
    tensor: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, TensorShardMetadata]:
    """Return this rank's flat, equally-sized shard and its reconstruction metadata."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")

    original_numel = tensor.numel()
    shard_numel = math.ceil(original_numel / world_size)
    metadata = TensorShardMetadata(
        original_shape=tensor.shape,
        original_numel=original_numel,
        shard_numel=shard_numel,
        world_size=world_size,
    )

    start = rank * shard_numel
    stop = min(start + shard_numel, original_numel)
    shard = tensor.new_zeros(shard_numel)
    if start < stop:
        shard[: stop - start].copy_(tensor.detach().reshape(-1)[start:stop])
    return shard, metadata


def _restore_full_tensor(
    padded_flat_tensor: torch.Tensor,
    metadata: TensorShardMetadata,
) -> torch.Tensor:
    """Remove shard padding and restore the original logical tensor shape."""
    if padded_flat_tensor.numel() != metadata.padded_numel:
        raise ValueError(
            "gathered tensor has "
            f"{padded_flat_tensor.numel()} elements, expected {metadata.padded_numel}"
        )
    return padded_flat_tensor[: metadata.original_numel].view(metadata.original_shape)
