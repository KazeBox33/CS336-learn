"""Educational fully-sharded data-parallel building blocks."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from cs336_basics.model import Embedding, Linear
from torch import nn


def _nvtx_range(
    message: str,
    tensor: torch.Tensor,
) -> AbstractContextManager[None]:
    if tensor.device.type == "cuda":
        return torch.cuda.nvtx.range(message)
    return nullcontext()


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
class PendingAllGather: # 保存正在进行的通信
    """Keep communication buffers alive until the gathered weight is consumed."""

    work: dist.Work
    input_shard: torch.Tensor
    output_buffer: torch.Tensor


@dataclass
class ShardedParameterState: # 记录一个被分片参数的全部管理信息
    """Track one sharded module weight and its persistent local master shard."""

    name: str
    module: nn.Module
    parameter: nn.Parameter
    local_shard: torch.Tensor
    metadata: TensorShardMetadata
    pending_all_gather: PendingAllGather | None = None


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
        self._pending_works: list[dist.Work] = [] # 通信句柄
        self._pending_reduce_scatter_inputs: list[torch.Tensor] = [] # 保存临时输入的buffer
        self._register_local_shards()
        self._register_replicated_gradient_hooks()
        self._forward_order: list[ShardedParameterState] = [] # 按需收集，记录顺序
        self._current_forward_order: list[ShardedParameterState] | None = None
        self._forward_order_matches = False

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
                child.register_forward_pre_hook(self._make_forward_pre_hook(state)) # 在该层计算前执行
            )
            self._hook_handles.append(
                child.register_forward_hook( # 在该层计算后执行
                    self._make_forward_post_hook(state),
                    always_call=True,
                )
            )
            self._hook_handles.append(
                child.register_full_backward_pre_hook(
                    self._make_backward_pre_hook(state)
                )
            )
            self._hook_handles.append(
                parameter.register_post_accumulate_grad_hook( # 它会在该参数完整梯度写入parameter.grad后执行
                    self._make_gradient_ready_hook(state)
                )
            )

    def _register_replicated_gradient_hooks(self) -> None:
        sharded_parameter_ids = {
            id(state.parameter) for state in self._sharded_parameter_states # 先收集已经分片的参数的id
        }
        for parameter in self.module.parameters():
            if parameter.requires_grad and id(parameter) not in sharded_parameter_ids: # 收集没有分片的参数
                self._hook_handles.append(
                    parameter.register_post_accumulate_grad_hook(
                        self._make_replicated_gradient_hook()
                    )
                )

    def _start_parameter_all_gather(
        self,
        state: ShardedParameterState,
        *,
        reason: str = "manual",
    ) -> None: # 进行 all gather 创建接收空间
        if state.pending_all_gather is not None:  #  如果有pending的，就不用 all gather 了，return 后直接 wait
            return

        communication_shard = state.local_shard # 先取出长期保存的FP32 shard
        if self.compute_dtype is not None:
            communication_shard = communication_shard.to(self.compute_dtype) # 变成bf16
        communication_shard = communication_shard.contiguous()

        gathered_flat_parameter = torch.empty( # 创建接收空间
            state.metadata.padded_numel,
            dtype=communication_shard.dtype,
            device=communication_shard.device,
        )
        with _nvtx_range(
            f"fsdp_all_gather_launch:{reason}:{state.name}",
            communication_shard,
        ):
            work = dist.all_gather_into_tensor(
                gathered_flat_parameter,
                communication_shard,
                async_op=True,
            )
        state.pending_all_gather = PendingAllGather(
            work=work,
            input_shard=communication_shard,
            output_buffer=gathered_flat_parameter,
        )

    def _materialize_full_parameter(
        self,
        state: ShardedParameterState,
        *,
        phase: str,
    ) -> None:
        self._start_parameter_all_gather(state, reason=f"{phase}_demand")
        pending = state.pending_all_gather
        assert pending is not None
        # Establish the communication dependency before accessing its output.
        with _nvtx_range(
            f"fsdp_all_gather_wait:{phase}:{state.name}",
            state.local_shard,
        ):
            pending.work.wait() # 等待 异步请求处理
        state.parameter.data = _restore_full_tensor(
            pending.output_buffer,
            state.metadata,
        )
        state.pending_all_gather = None

    @staticmethod
    def _restore_local_parameter(state: ShardedParameterState) -> None:
        state.parameter.data = state.local_shard

    def _reduce_scatter_parameter_gradient(
        self,
        state: ShardedParameterState,
    ) -> None:
        full_gradient = state.parameter.grad
        if full_gradient is None:
            raise RuntimeError(f"gradient for {state.name} is not available")
        if full_gradient.numel() != state.metadata.original_numel: # 说明此时的参数是完整的
            raise RuntimeError(
                f"gradient for {state.name} has {full_gradient.numel()} elements, "
                f"expected {state.metadata.original_numel}"
            )

        padded_gradient = state.local_shard.new_zeros(state.metadata.padded_numel)
        padded_gradient[: state.metadata.original_numel].copy_(
            full_gradient.detach().reshape(-1)
        )
        padded_gradient.div_(self.world_size)

        local_gradient = torch.empty_like(state.local_shard)
        work = dist.reduce_scatter_tensor(
            local_gradient,
            padded_gradient,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        state.parameter.grad = None
        self._restore_local_parameter(state)
        state.parameter.grad = local_gradient
        self._pending_works.append(work)
        self._pending_reduce_scatter_inputs.append(padded_gradient)

    def _make_forward_pre_hook(
        self,
        state: ShardedParameterState,
    ) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def gather_full_parameter(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            trace = self._current_forward_order
            index = len(trace) if trace is not None else 0
            if self._forward_order_matches and (
                index >= len(self._forward_order)
                or self._forward_order[index] is not state # 检查当前执行层是否符合上一层的记录 ， 不符合就采取下面的措施进行清理
            ):
                self._discard_unused_all_gathers()
                self._forward_order_matches = False
            self._materialize_full_parameter(state, phase="forward")
            if trace is not None:
                trace.append(state) # current 记录完整L1
                if self._forward_order_matches and index + 1 < len(self._forward_order):
                    self._start_parameter_all_gather(
                        self._forward_order[index + 1],
                        reason="forward_prefetch",
                    )  # 这一步只发起 all gather 没有 wait
            if state.local_shard.device.type == "cuda":
                torch.cuda.nvtx.range_push(f"fsdp_forward_compute:{state.name}")

        return gather_full_parameter

    def _make_forward_post_hook(
        self,
        state: ShardedParameterState,
    ) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def free_full_parameter(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> Any:
            if state.local_shard.device.type == "cuda":
                torch.cuda.nvtx.range_pop()
            self._restore_local_parameter(state)
            return output

        return free_full_parameter

    def _make_backward_pre_hook(
        self,
        state: ShardedParameterState,
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def gather_full_parameter(
            _module: nn.Module,
            _grad_outputs: tuple[torch.Tensor, ...],
        ) -> None:
            self._materialize_full_parameter(state, phase="backward")

        return gather_full_parameter

    def _make_gradient_ready_hook(
        self,
        state: ShardedParameterState,
    ) -> Callable[[torch.Tensor], None]:
        def reduce_scatter_gradient(_parameter: torch.Tensor) -> None:
            self._reduce_scatter_parameter_gradient(state)

        return reduce_scatter_gradient

    def _make_replicated_gradient_hook(
        self,
    ) -> Callable[[torch.Tensor], None]:
        def synchronize_gradient(parameter: torch.Tensor) -> None:
            if parameter.grad is None:
                raise RuntimeError("replicated parameter gradient is not available")

            with torch.no_grad():
                parameter.grad.div_(self.world_size) # 核心部分，梯度除以world_size
                work = dist.all_reduce( # 再进行all reduce
                    parameter.grad,
                    op=dist.ReduceOp.SUM,
                    async_op=True, # 这里异步可以为 computation/communication overlap 提供可能
                )
            self._pending_works.append(work)

        return synchronize_gradient

    def finish_gradient_synchronization(self) -> None: # 需要在 optimizer 前梯度更新完
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()
        self._pending_reduce_scatter_inputs.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:  # 为了保留完整快照 ， 测试 optimizer.step 后的参数
        """Collect independent master-weight snapshots; every rank must call this."""
        self.finish_gradient_synchronization()
        states_by_id = {
            id(state.parameter): state for state in self._sharded_parameter_states
        }
        full_params = {}
        for name, parameter in self.module.named_parameters():
            state = states_by_id.get(id(parameter))
            if state is None:
                full_params[name] = parameter.detach().clone()
                continue

            gathered = state.local_shard.new_empty(state.metadata.padded_numel)
            dist.all_gather_into_tensor(gathered, state.local_shard.contiguous())
            full_params[name] = _restore_full_tensor(gathered, state.metadata).clone()
        return full_params

    def _discard_unused_all_gathers(self) -> None:  # 清理预存的
        for state in self._sharded_parameter_states:
            pending = state.pending_all_gather
            if pending is not None:
                pending.work.wait()
                state.pending_all_gather = None

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        """Learn execution order, then prefetch one sharded layer ahead."""
        self._current_forward_order = []
        self._forward_order_matches = bool(self._forward_order) # 第二层开始这里是 True
        try:
            output = self.module(*inputs, **kwargs)
            self._forward_order = self._current_forward_order # 保存顺序然后替换
            return output
        finally:
            # A shortened path or exception may leave an unused prefetch in flight.
            self._discard_unused_all_gathers()
            self._current_forward_order = None
            self._forward_order_matches = False
