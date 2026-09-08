import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .adapters import get_ddp
from .common import ToyModel, _cleanup_process_group, _setup_process_group, validate_ddp_net_equivalence


def test_naive_ddp_broadcasts_state_and_forwards() -> None:
    world_size = 2
    mp.spawn(_test_naive_ddp_broadcasts_state_and_forwards, args=(world_size,), nprocs=world_size, join=True)


def _test_naive_ddp_broadcasts_state_and_forwards(rank: int, world_size: int) -> None:
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        torch.manual_seed(rank)
        module = ToyModel().to(device)
        ddp_model = get_ddp(module)

        assert ddp_model.module is module
        validate_ddp_net_equivalence(ddp_model)

        inputs = torch.ones(2, 10, device=device)
        assert torch.equal(ddp_model(inputs), module(inputs))
        dist.barrier()
    finally:
        if dist.is_initialized():
            _cleanup_process_group()
