"""Incremental implementation of an optimizer with sharded state."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


def _assign_parameter_owners(
    parameters: Iterable[torch.Tensor],
    assigned_numels: list[int],  # 每个 rank 已经被分配了多少
) -> list[int]:  # list 记录每个参数的 owner
    """Return one owner per tensor and update the per-rank element counts.

    Every rank must supply the same parameter order and starting counts.
    """
    if not assigned_numels or any(count < 0 for count in assigned_numels):
        raise ValueError("assigned_numels must contain non-negative counts for each rank")

    owners = []
    for parameter in parameters:
        owner = min(range(len(assigned_numels)), key=assigned_numels.__getitem__)
        owners.append(owner)
        assigned_numels[owner] += parameter.numel()
    return owners


class ShardedOptimizer(Optimizer):
    """Shard optimizer state and synchronize updated parameters across ranks."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[Optimizer],  # 真正局部执行的 Optimizer
        **kwargs: Any,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer requires an initialized process group")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._assigned_numels = [0] * self.world_size
        self._parameter_owners: list[tuple[torch.Tensor, int]] = []  # 记录参数的 owner
        # 适配 PyTorch optimizer，用于保存参数和对应的超参数。

        # 本地参数
        self._local_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: Optimizer | None = None

        # Optimizer.__init__ calls our add_param_group before the local optimizer exists.
        super().__init__(params, defaults=kwargs)  # 父类构造会调用重写的 add_param_group
        self._local_optimizer = optimizer_cls(self._local_param_groups, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(dict(param_group))  # 浅拷贝
        group = self.param_groups[-1]
        owners = _assign_parameter_owners(group["params"], self._assigned_numels)
        # zip 把两个列表按位置组合，例如 (p0, 0)。
        self._parameter_owners.extend(zip(group["params"], owners, strict=True))

        local_group = dict(group)
        local_group["params"] = [
            parameter
            for parameter, owner in zip(group["params"], owners, strict=True)
            if owner == self.rank
        ]
        # Keep empty groups so group indices and hyperparameters match on every rank.
        self._local_param_groups.append(local_group)
        if self._local_optimizer is not None:
            self._local_optimizer.add_param_group(local_group)

    def step(self, closure: Any = None, **kwargs: Any) -> Any:
        """Update the local shard, then synchronize every parameter from its owner."""
        if self._local_optimizer is None:
            raise RuntimeError("local optimizer has not been initialized")

        loss = self._local_optimizer.step(closure=closure, **kwargs)
        with torch.no_grad():
            for parameter, owner in self._parameter_owners:
                dist.broadcast(parameter, src=owner) # 遍历并广播
        return loss
