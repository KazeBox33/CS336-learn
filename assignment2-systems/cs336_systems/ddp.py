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
        if not dist.is_initialized():
            raise RuntimeError("NaiveDDP requires an initialized process group")

        self.module = module
        self._broadcast_module_state()

    def _broadcast_module_state(self) -> None:
        """Copy rank 0's parameters and buffers to every process."""
        with torch.no_grad():
            for tensor in self.module.state_dict().values():
                dist.broadcast(tensor, src=0)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)
