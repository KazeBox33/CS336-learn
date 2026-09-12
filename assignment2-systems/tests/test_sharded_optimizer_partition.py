import torch

from cs336_systems.sharded_optimizer import _assign_parameter_owners


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
