from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cs336_systems.checkpointing import (
    checkpoint_layers_in_segments,
    recursive_checkpoint_layers,
    run_layers,
)


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.gelu(self.linear(x))


def run_and_collect_gradients(
    layers: nn.ModuleList,
    x: torch.Tensor,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    if strategy == "none":
        output = run_layers(layers, x)
    elif strategy == "segment":
        output = checkpoint_layers_in_segments(layers, x, segment_size=2)
    else:
        output = recursive_checkpoint_layers(layers, x)

    output.square().mean().backward()
    parameter_gradients = [
        parameter.grad.detach().clone()
        for layer in layers
        for parameter in layer.parameters()
    ]
    return output.detach(), x.grad.detach().clone(), parameter_gradients


@pytest.mark.parametrize("strategy", ["segment", "recursive"])
def test_checkpointing_matches_uncheckpointed_gradients(strategy: str) -> None:
    torch.manual_seed(0)
    baseline_layers = nn.ModuleList([ResidualBlock(8) for _ in range(4)])
    checkpointed_layers = copy.deepcopy(baseline_layers)

    baseline_x = torch.randn(2, 3, 8, requires_grad=True)
    checkpointed_x = baseline_x.detach().clone().requires_grad_(True)

    baseline = run_and_collect_gradients(baseline_layers, baseline_x, "none")
    checkpointed = run_and_collect_gradients(
        checkpointed_layers,
        checkpointed_x,
        strategy,
    )

    torch.testing.assert_close(checkpointed[0], baseline[0])
    torch.testing.assert_close(checkpointed[1], baseline[1])
    for actual, expected in zip(checkpointed[2], baseline[2], strict=True):
        torch.testing.assert_close(actual, expected)


def test_segment_checkpoint_rejects_nonpositive_segment_size() -> None:
    x = torch.ones(1, requires_grad=True)

    with pytest.raises(ValueError, match="segment_size must be positive"):
        checkpoint_layers_in_segments([nn.Identity()], x, segment_size=0)
