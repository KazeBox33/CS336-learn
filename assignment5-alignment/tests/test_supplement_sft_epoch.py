from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.supplement.data import PackedSFTDataset, iterate_batches
from cs336_alignment.supplement.sft import compute_sft_loss, run_sft_epoch
from tests.common import FIXTURES_PATH


def make_batches():
    batches = []
    for index, size in enumerate((2, 2, 2, 2, 1)):
        tokens = (torch.arange(size * 4).reshape(size, 4) + index) % 10 + 3
        batches.append({"input_ids": tokens, "labels": (tokens + 1) % 15})
    return batches


@pytest.mark.parametrize("accumulation_steps", [1, 2, 5, 8])
def test_sft_epoch_matches_sequential_full_batch_updates(tiny_train_model, accumulation_steps):
    batches = make_batches()
    reference = deepcopy(tiny_train_model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    expected = []
    for start in range(0, len(batches), accumulation_steps):
        group = batches[start:start + accumulation_steps]
        full_batch = {key: torch.cat([batch[key] for batch in group]) for key in group[0]}
        reference_optimizer.zero_grad(set_to_none=True)
        loss = compute_sft_loss(reference, full_batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        reference_optimizer.step()
        expected.append((loss.item(), norm.item(), full_batch["labels"].numel(), len(group)))

    optimizer = torch.optim.AdamW(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    results = list(run_sft_epoch(tiny_train_model, optimizer, iter(batches), accumulation_steps, 0.1))

    assert optimizer.step.call_count == len(expected)
    assert [update for update, _ in results] == list(range(1, len(expected) + 1))
    for (_, metrics), (loss, norm, tokens, count) in zip(results, expected, strict=True):
        assert metrics["loss"] == pytest.approx(loss, rel=1e-5)
        assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-5)
        assert metrics["num_tokens"] == tokens
        assert metrics["num_microbatches"] == count
        assert all(isinstance(value, (int, float)) for value in metrics.values())
    for actual, target in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, target, rtol=1e-5, atol=1e-6)
        assert actual.grad is None


def test_sft_epoch_consumes_only_one_group_per_request(tiny_train_model):
    consumed = []

    def source():
        for index, batch in enumerate(make_batches()):
            consumed.append(index)
            yield batch

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    updates = run_sft_epoch(tiny_train_model, optimizer, source(), 2)
    assert consumed == []
    optimizer.step.assert_not_called()
    for update, expected_indices in enumerate(([0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]), start=1):
        step, metrics = next(updates)
        assert step == update
        assert consumed == expected_indices
        assert optimizer.step.call_count == update
        assert metrics["num_microbatches"] == (1 if update == 3 else 2)
    with pytest.raises(StopIteration):
        next(updates)
    assert optimizer.step.call_count == 3


@pytest.mark.parametrize("accumulation_steps", [0, -1, 1.5])
def test_sft_epoch_rejects_invalid_accumulation_before_reading_data(tiny_train_model, accumulation_steps):
    consumed = []

    def source():
        for batch in make_batches():
            consumed.append(batch)
            yield batch

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    with pytest.raises(ValueError, match="positive integer"):
        list(run_sft_epoch(tiny_train_model, optimizer, source(), accumulation_steps))
    assert consumed == []
    optimizer.step.assert_not_called()


def test_empty_sft_epoch_does_not_update(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    assert list(run_sft_epoch(tiny_train_model, optimizer, [], 2)) == []
    optimizer.step.assert_not_called()


def test_sft_epoch_covers_every_packed_sample_once(tiny_train_model, tokenizer):
    dataset = PackedSFTDataset(tokenizer, FIXTURES_PATH / "sft_sample.jsonl", seq_length=8, shuffle=False)
    loader = iterate_batches(dataset, batch_size=7, shuffle=False)
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    results = list(run_sft_epoch(tiny_train_model, optimizer, loader, 3))
    assert len(results) == (len(loader) + 2) // 3
    assert sum(metrics["num_microbatches"] for _, metrics in results) == len(loader)
    assert sum(metrics["num_tokens"] for _, metrics in results) == len(dataset) * 8
