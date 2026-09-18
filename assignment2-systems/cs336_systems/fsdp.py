"""Educational fully-sharded data-parallel building blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear
from torch import nn


@dataclass(frozen=True)
class TensorShardMetadata: # 元数据
    """Describe how one logical tensor is flattened and divided across ranks."""

    original_shape: torch.Size # 参数原来的shape
    original_numel: int # padding前的真实元素数
    shard_numel: int # 每个rank 保存多少元素
    world_size: int

    @property # 让属性像方法一样使用
    def padded_numel(self) -> int:
        return self.shard_numel * self.world_size


@dataclass
class ShardedParameterState: # 记录一个被分片参数的全部管理信息
    """Track one sharded module weight and its persistent local master shard."""

    name: str
    module: nn.Module
    parameter: nn.Parameter
    local_shard: torch.Tensor
    metadata: TensorShardMetadata


def _shard_tensor(
    tensor: torch.Tensor,
    *, # 表示后面的必须写出名字
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
    metadata = TensorShardMetadata( # 以后用来恢复的元数据
        original_shape=tensor.shape,
        original_numel=original_numel,
        shard_numel=shard_numel,
        world_size=world_size,
    )

    start = rank * shard_numel
    stop = min(start + shard_numel, original_numel)
    shard = tensor.new_zeros(shard_numel)
    if start < stop:
        shard[: stop - start].copy_(tensor.detach().reshape(-1)[start:stop]) # detach 是因为不应连接到原参数的autograd gragh
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
    return padded_flat_tensor[: metadata.original_numel].view(metadata.original_shape) # 去掉 padding


class FullyShardedDataParallel(nn.Module):
    """Own local FP32 shards for Linear and Embedding weights."""

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("FullyShardedDataParallel requires an initialized process group")

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._sharded_parameter_states: list[ShardedParameterState] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_local_shards()

    def _register_local_shards(self) -> None:
        for module_name, child in self.module.named_modules():
            if not isinstance(child, (Linear, Embedding)): # 不是Linear 或者 Embedding 就过滤
                continue

            parameter = child.weight
            local_shard, metadata = _shard_tensor( # 分出 shard
                parameter,
                rank=self.rank,
                world_size=self.world_size,
            )
            parameter.data = local_shard # 关键一步，替换掉了 .data
            parameter_name = f"{module_name}.weight" if module_name else "weight"
            state = ShardedParameterState(
                name=parameter_name,
                module=child,
                parameter=parameter,
                local_shard=local_shard,
                metadata=metadata,
            )
            self._sharded_parameter_states.append(state)
            self._hook_handles.append(
                child.register_forward_pre_hook(self._make_forward_pre_hook(state))
            )
            self._hook_handles.append(
                child.register_forward_hook(
                    self._make_forward_post_hook(state),
                    always_call=True,
                )
            )

    def _materialize_full_parameter(self, state: ShardedParameterState) -> None:
        communication_shard = state.local_shard
        if self.compute_dtype is not None:
            communication_shard = communication_shard.to(self.compute_dtype)

        gathered_flat_parameter = torch.empty(
            state.metadata.padded_numel,
            dtype=communication_shard.dtype,
            device=communication_shard.device,
        )
        dist.all_gather_into_tensor(
            gathered_flat_parameter,
            communication_shard.contiguous(),
        )
        state.parameter.data = _restore_full_tensor(
            gathered_flat_parameter,
            state.metadata,
        )

    @staticmethod
    def _restore_local_parameter(state: ShardedParameterState) -> None:
        state.parameter.data = state.local_shard

    def _make_forward_pre_hook(self, state: ShardedParameterState):
        def gather_full_parameter(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._materialize_full_parameter(state)

        return gather_full_parameter

    def _make_forward_post_hook(self, state: ShardedParameterState):
        def free_full_parameter(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            self._restore_local_parameter(state)
            return output

        return free_full_parameter

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)
