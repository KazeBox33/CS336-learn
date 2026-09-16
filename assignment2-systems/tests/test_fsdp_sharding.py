import pytest
import torch
from cs336_basics.model import Embedding, Linear, RMSNorm
from torch import nn

from cs336_systems.fsdp import (
    FullyShardedDataParallel,
    _restore_full_tensor,
    _shard_tensor,
)


def test_shard_tensor_splits_divisible_tensor_into_equal_flat_chunks() -> None:
    tensor = torch.arange(12, dtype=torch.float32).view(3, 4)

    shard_0, metadata_0 = _shard_tensor(tensor, rank=0, world_size=2)
    shard_1, metadata_1 = _shard_tensor(tensor, rank=1, world_size=2)

    torch.testing.assert_close(shard_0, torch.arange(6, dtype=torch.float32))
    torch.testing.assert_close(shard_1, torch.arange(6, 12, dtype=torch.float32))
    assert metadata_0 == metadata_1
    assert metadata_0.original_shape == torch.Size([3, 4])
    assert metadata_0.shard_numel == 6
    assert metadata_0.padded_numel == 12


def test_shard_tensor_detaches_parameter_history() -> None:
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))

    shard, _ = _shard_tensor(parameter, rank=0, world_size=2)

    assert not shard.requires_grad
    assert shard.grad_fn is None


def test_shard_tensor_pads_last_rank_and_restores_original_shape() -> None:
    tensor = torch.arange(5, dtype=torch.float32).view(5, 1)

    shard_0, metadata = _shard_tensor(tensor, rank=0, world_size=2)
    shard_1, _ = _shard_tensor(tensor, rank=1, world_size=2)

    torch.testing.assert_close(shard_0, torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(shard_1, torch.tensor([3.0, 4.0, 0.0]))
    restored = _restore_full_tensor(torch.cat([shard_0, shard_1]), metadata)
    torch.testing.assert_close(restored, tensor)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 2), (2, 2), (0, 0)],
)
def test_shard_tensor_rejects_invalid_process_coordinates(rank: int, world_size: int) -> None:
    with pytest.raises(ValueError):
        _shard_tensor(torch.ones(4), rank=rank, world_size=world_size)


def test_restore_full_tensor_rejects_wrong_gathered_size() -> None:
    _, metadata = _shard_tensor(torch.ones(5), rank=0, world_size=2)

    with pytest.raises(ValueError, match="expected 6"):
        _restore_full_tensor(torch.ones(5), metadata)


class _ShardableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = Embedding(5, 1)
        self.linear = Linear(1, 4)
        self.norm = RMSNorm(4)


def _mock_process_group(monkeypatch: pytest.MonkeyPatch, *, rank: int) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)


def test_fsdp_constructor_registers_local_master_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_process_group(monkeypatch, rank=1)
    model = _ShardableModel()
    with torch.no_grad():
        model.embedding.weight.copy_(torch.arange(5, dtype=torch.float32).view(5, 1))
    embedding_parameter = model.embedding.weight
    norm_shape = model.norm.weight.shape

    fsdp = FullyShardedDataParallel(model, compute_dtype=torch.float16)

    assert fsdp.compute_dtype == torch.float16
    assert model.embedding.weight is embedding_parameter
    assert model.embedding.weight.dtype == torch.float32
    torch.testing.assert_close(model.embedding.weight, torch.tensor([3.0, 4.0, 0.0]))
    assert model.linear.weight.shape == torch.Size([2])
    assert model.norm.weight.shape == norm_shape
    assert [state.name for state in fsdp._sharded_parameter_states] == [
        "embedding.weight",
        "linear.weight",
    ]
    assert all(state.parameter is state.module.weight for state in fsdp._sharded_parameter_states)


def test_fsdp_constructor_requires_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="initialized process group"):
        FullyShardedDataParallel(_ShardableModel())
