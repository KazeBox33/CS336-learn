from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.supplement import sft
from cs336_alignment.supplement.sft import create_sft_lr_scheduler, run_sft_epoch


def make_batches(count):
    for _ in range(count):
        yield {"input_ids": torch.tensor([[3, 4, 5, 6]]), "labels": torch.tensor([[4, 5, 6, 7]])}


def test_sft_schedule_linear_warmup_cosine_decay_and_group_scaling():
    parameters = [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))]
    optimizer = torch.optim.SGD([
        {"params": [parameters[0]], "lr": 0.2},
        {"params": [parameters[1]], "lr": 0.02},
    ])
    scheduler = create_sft_lr_scheduler(optimizer, total_updates=5, warmup_updates=2)
    for factor in [0.0, 0.5, 1.0, 0.75, 0.25, 0.0, 0.0]:
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.2 * factor, 0.02 * factor])
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize("total_updates", [1, 4])
def test_sft_schedule_without_warmup_starts_at_peak(total_updates):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = create_sft_lr_scheduler(optimizer, total_updates=total_updates)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    for _ in range(total_updates):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


@pytest.mark.parametrize("total, warmup", [(0, 0), (-1, 0), (2.5, 0), (4, -1), (4, 4), (4, 5), (4, 1.5)])
def test_sft_schedule_rejects_invalid_counts(total, warmup):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    with pytest.raises(ValueError):
        create_sft_lr_scheduler(optimizer, total, warmup)
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_sft_schedule_advances_once_per_update_including_tail(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    scheduler = create_sft_lr_scheduler(optimizer, total_updates=3)
    scheduler.step = Mock(wraps=scheduler.step)
    used_rates = []
    optimizer.register_step_pre_hook(lambda opt, args, kwargs: used_rates.append(opt.param_groups[0]["lr"]))
    results = list(run_sft_epoch(tiny_train_model, optimizer, make_batches(5), 2, scheduler=scheduler))
    assert scheduler.step.call_count == 3
    assert [metrics["num_microbatches"] for _, metrics in results] == [2, 2, 1]
    assert used_rates == pytest.approx([0.1, 0.075, 0.025])
    assert [metrics["learning_rate"] for _, metrics in results] == pytest.approx(used_rates)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_sft_schedule_continues_across_epochs(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    scheduler = create_sft_lr_scheduler(optimizer, total_updates=4, warmup_updates=1)
    results = []
    for _ in range(2):
        results.extend(run_sft_epoch(tiny_train_model, optimizer, make_batches(3), 2, scheduler=scheduler))
    assert [step for step, _ in results] == [1, 2, 1, 2]
    assert [metrics["learning_rate"] for _, metrics in results] == pytest.approx([0.0, 0.1, 0.075, 0.025])
    assert scheduler.last_epoch == 4


def test_sft_epoch_without_scheduler_keeps_learning_rate(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    results = list(run_sft_epoch(tiny_train_model, optimizer, make_batches(3), 2))
    assert [metrics["learning_rate"] for _, metrics in results] == [0.1, 0.1]
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_sft_empty_epoch_does_not_advance_schedule(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    scheduler = create_sft_lr_scheduler(optimizer, total_updates=3)
    assert list(run_sft_epoch(tiny_train_model, optimizer, [], 2, scheduler=scheduler)) == []
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_sft_failed_update_does_not_advance_schedule(tiny_train_model, monkeypatch):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    scheduler = create_sft_lr_scheduler(optimizer, total_updates=3)
    monkeypatch.setattr(sft, "sft_train_step", Mock(side_effect=RuntimeError("update failed")))
    with pytest.raises(RuntimeError, match="update failed"):
        list(run_sft_epoch(tiny_train_model, optimizer, make_batches(2), 2, scheduler=scheduler))
    assert scheduler.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_sft_epoch_rejects_scheduler_for_another_optimizer(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    other_optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    scheduler = create_sft_lr_scheduler(other_optimizer, total_updates=3)
    with pytest.raises(ValueError, match="belong"):
        list(run_sft_epoch(tiny_train_model, optimizer, make_batches(2), 2, scheduler=scheduler))
    assert scheduler.last_epoch == 0
