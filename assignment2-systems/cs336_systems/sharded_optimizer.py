"""Parameter ownership helpers for optimizer state sharding."""

from collections.abc import Iterable

import torch


def _assign_parameter_owners(
    parameters: Iterable[torch.Tensor],
    assigned_numels: list[int],
) -> list[int]:
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
