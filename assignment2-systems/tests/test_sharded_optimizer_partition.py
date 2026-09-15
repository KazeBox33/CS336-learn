import pytest
import torch

from cs336_systems.sharded_optimizer import ShardedOptimizer, _assign_parameter_owners


def test_ownership_balances_element_counts_deterministically() -> None:
    parameters = [torch.empty(size) for size in (8, 2, 6, 4, 4)]
    counts = [0, 0]
    owners = _assign_parameter_owners(iter(parameters), counts)

    assert owners == [0, 1, 1, 0, 1]
    assert counts == [12, 12]
    assert _assign_parameter_owners(parameters, [0, 0]) == owners


def test_new_group_uses_existing_rank_loads() -> None:
    counts = [0, 0]
    first_owners = _assign_parameter_owners([torch.empty(8), torch.empty(2)], counts)
    new_owners = _assign_parameter_owners([torch.empty(6)], counts)

    assert first_owners == [0, 1]
    assert new_owners == [1]
    assert counts == [8, 8]


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_constructor_preserves_groups_and_clears_all_gradients(monkeypatch, rank) -> None:
    # Only ownership depends on rank here; construction performs no collectives.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)
    parameters = [torch.nn.Parameter(torch.zeros(size)) for size in (8, 2)]
    groups = [
        {"params": [parameters[0]], "lr": 0.01},
        {"params": [parameters[1]], "lr": 0.02},
    ]

    optimizer = ShardedOptimizer(groups, torch.optim.AdamW, weight_decay=0.1)
    local_groups = optimizer._local_optimizer.param_groups
    assert [group["lr"] for group in local_groups] == [0.01, 0.02]
    assert all(group["weight_decay"] == 0.1 for group in local_groups)
    assert [id(p) for group in local_groups for p in group["params"]] == (
        [id(parameters[rank])] if rank < 2 else []
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer._assigned_numels == [8, 2, 0]

    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in parameters)

    new_parameter = torch.nn.Parameter(torch.zeros(6))
    optimizer.add_param_group({"params": [new_parameter], "lr": 0.03})
    assert optimizer._assigned_numels == [8, 2, 6]
    assert local_groups[-1]["lr"] == 0.03
    assert [id(p) for p in local_groups[-1]["params"]] == (
        [id(new_parameter)] if rank == 2 else []
    )


def test_constructor_requires_process_group(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized process group"):
        ShardedOptimizer([torch.nn.Parameter(torch.zeros(1))], torch.optim.AdamW)


def test_step_updates_local_parameters_and_broadcasts_all_parameters(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    broadcast_calls = []
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda parameter, src: broadcast_calls.append((parameter, src)),
    )

    parameters = [
        torch.nn.Parameter(torch.tensor([1.0])),
        torch.nn.Parameter(torch.tensor([2.0, 2.0])),
    ]
    optimizer = ShardedOptimizer(parameters, torch.optim.SGD, lr=0.1)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    torch.testing.assert_close(parameters[0], torch.tensor([0.9]))
    torch.testing.assert_close(parameters[1], torch.tensor([2.0, 2.0]))
    assert [(id(parameter), src) for parameter, src in broadcast_calls] == [
        (id(parameters[0]), 0),
        (id(parameters[1]), 1),
    ]
