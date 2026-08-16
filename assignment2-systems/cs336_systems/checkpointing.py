from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint #导入checkpoint

TensorLayer = Callable[[torch.Tensor], torch.Tensor] #类型别名，接收一个torch.Tensor 返回一个torch.Tensor


def run_layers(
    layers: Sequence[TensorLayer], #Sequence表示一个有顺序、可以遍历的容器 里面是 TensorLayer
    x: torch.Tensor,
) -> torch.Tensor:
    """Run a sequence of layers without activation checkpointing."""
    for layer in layers:
        x = layer(x)
    return x


def checkpoint_layers_in_segments( #按照段来设置checkpoint
    layers: Sequence[TensorLayer],
    x: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """Checkpoint consecutive, non-nested segments of ``segment_size`` layers."""
    if segment_size <= 0:
        raise ValueError("segment_size must be positive")

    layers = tuple(layers)
    if not layers:
        return x

    for start in range(0, len(layers), segment_size):
        segment = layers[start : start + segment_size]
        x = checkpoint(
            partial(run_layers, segment), # partial 是用来固定参数的，相当于把segement 这个参数给固定了
            x,
            use_reentrant=False,
        )

    return x


def recursive_checkpoint_layers(
    layers: Sequence[TensorLayer],
    x: torch.Tensor,
) -> torch.Tensor:
    """Recursively bisect layers and nest non-reentrant checkpoints."""
    layers = tuple(layers)
    if not layers:
        return x

    return _recursive_checkpoint_layers(layers, x)


def _recursive_checkpoint_layers(
    layers: tuple[TensorLayer, ...],
    x: torch.Tensor,
) -> torch.Tensor:
    if len(layers) == 1:
        return layers[0](x)

    middle = len(layers) // 2

    def run_segment(segment_input: torch.Tensor) -> torch.Tensor:
        hidden = _recursive_checkpoint_layers(layers[:middle], segment_input)
        return _recursive_checkpoint_layers(layers[middle:], hidden)

    return checkpoint(run_segment, x, use_reentrant=False)
