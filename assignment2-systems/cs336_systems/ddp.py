"""Educational distributed-data-parallel containers."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    """Wrap a module and initialize identical model state on every rank."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_initialized():  # 先检查一下有没有初始化
            raise RuntimeError("NaiveDDP requires an initialized process group")

        self.module = module
        self._broadcast_module_state()

    def _broadcast_module_state(self) -> None:  # 广播种子
        """Copy rank 0's parameters and buffers to every process."""
        with torch.no_grad():  # 初始化同步不是训练计算，不需要建立 autograd graph
            for tensor in self.module.state_dict().values():  # state_dict 不只有参数，还有 buffer
                dist.broadcast(tensor, src=0)  # 以 rank 0 为源广播

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:  # 梯度同步
        """Average every available parameter gradient across all ranks."""
        world_size = dist.get_world_size()
        with torch.no_grad():
            for parameter in self.module.parameters():
                if parameter.grad is None:
                    continue
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)


class FlatGradientDDP(NaiveDDP):
    """Synchronize all dense parameter gradients with one all-reduce."""

    def finish_gradient_synchronization(self) -> None:
        gradients = [
            parameter.grad
            for parameter in self.module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return

        reference_gradient = gradients[0]
        if any(
            gradient.device != reference_gradient.device
            or gradient.dtype != reference_gradient.dtype
            for gradient in gradients[1:]
        ):
            raise RuntimeError(
                "FlatGradientDDP requires all gradients to have the same device and dtype"
            )

        with torch.no_grad():
            flat_gradient = torch.cat(
                [gradient.reshape(-1) for gradient in gradients]
            )
            dist.all_reduce(flat_gradient, op=dist.ReduceOp.SUM)
            flat_gradient.div_(dist.get_world_size())

            offset = 0
            for gradient in gradients:
                numel = gradient.numel()
                gradient.copy_(
                    flat_gradient[offset : offset + numel].view_as(gradient)
                )
                offset += numel
