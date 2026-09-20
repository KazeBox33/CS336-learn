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


@pytest.mark.parametrize(
    ("compute_dtype", "expected_output_dtype"),
    [(None, torch.float32), (torch.float16, torch.float16)],
)
def test_fsdp_forward_gathers_full_weight_then_restores_local_shard(
    monkeypatch: pytest.MonkeyPatch,
    compute_dtype: torch.dtype | None,
    expected_output_dtype: torch.dtype,
) -> None:
    _mock_process_group(monkeypatch, rank=0)
    embedding = Embedding(5, 1)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(5, dtype=torch.float32).view(5, 1))

    gathered_input_dtypes = []

    def fake_all_gather_into_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> None:
        gathered_input_dtypes.append(input_tensor.dtype)
        padded_full_weight = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
            dtype=input_tensor.dtype,
        )
        output_tensor.copy_(padded_full_weight)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    fsdp = FullyShardedDataParallel(embedding, compute_dtype=compute_dtype)
    state = fsdp._sharded_parameter_states[0]

    output = fsdp(torch.tensor([0, 4]))

    torch.testing.assert_close(
        output,
        torch.tensor([[0.0], [4.0]], dtype=expected_output_dtype),
    )
    assert gathered_input_dtypes == [expected_output_dtype]
    assert embedding.weight.dtype == torch.float32
    assert embedding.weight.shape == torch.Size([3])
    assert embedding.weight.data_ptr() == state.local_shard.data_ptr()
    torch.testing.assert_close(embedding.weight, torch.tensor([0.0, 1.0, 2.0]))


def test_fsdp_backward_gathers_weight_and_reduce_scatters_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_process_group(monkeypatch, rank=0)
    linear = Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    all_gather_calls = []
    reduce_scatter_inputs = []

    def fake_all_gather_into_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> None:
        all_gather_calls.append(input_tensor.clone())
        output_tensor.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def fake_reduce_scatter_tensor(
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        *,
        op: torch.distributed.ReduceOp,
    ) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        reduce_scatter_inputs.append(input_tensor.clone())
        output_tensor.copy_(input_tensor[: output_tensor.numel()] * 2)

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        fake_all_gather_into_tensor,
    )
    monkeypatch.setattr(
        torch.distributed,
        "reduce_scatter_tensor",
        fake_reduce_scatter_tensor,
    )
    fsdp = FullyShardedDataParallel(linear)
    state = fsdp._sharded_parameter_states[0]
    inputs = torch.tensor([[1.0, 1.0]], requires_grad=True)

    output = fsdp(inputs)
    output.sum().backward()

    assert len(all_gather_calls) == 2
    torch.testing.assert_close(inputs.grad, torch.tensor([[4.0, 6.0]]))
    torch.testing.assert_close(
        reduce_scatter_inputs[0],
        torch.full((4,), 0.5),
    )
    assert linear.weight.data_ptr() == state.local_shard.data_ptr()
    torch.testing.assert_close(linear.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(linear.weight.grad, torch.tensor([1.0, 1.0]))


def test_fsdp_synchronizes_replicated_gradients_asynchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_process_group(monkeypatch, rank=0)
    model = _ShardableModel()
    fsdp = FullyShardedDataParallel(model)
    replicated_parameter = model.norm.weight
    replicated_parameter.grad = torch.tensor([2.0, 4.0, 6.0, 8.0])
    all_reduce_inputs = []
    waited = []

    class FakeWork:
        def wait(self) -> None:
            waited.append(True)

    def fake_all_reduce(
        tensor: torch.Tensor,
        *,
        op: torch.distributed.ReduceOp,
        async_op: bool,
    ) -> FakeWork:
        assert op == torch.distributed.ReduceOp.SUM
        assert async_op
        all_reduce_inputs.append(tensor.clone())
        tensor.mul_(2)
        return FakeWork()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    fsdp._make_replicated_gradient_hook()(replicated_parameter)

    torch.testing.assert_close(
        all_reduce_inputs[0],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    fsdp.finish_gradient_synchronization()

    assert waited == [True]
    assert fsdp._pending_works == []
    torch.testing.assert_close(
        replicated_parameter.grad,
        torch.tensor([2.0, 4.0, 6.0, 8.0]),
    )
