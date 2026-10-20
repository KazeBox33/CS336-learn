import math
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.supplement import sft
from cs336_alignment.supplement.data import PackedSFTDataset, iterate_batches
from cs336_alignment.supplement.sft import compute_sft_loss, sft_train_step
from tests.common import FIXTURES_PATH


def make_batches(sizes):
    batch_size = sum(sizes)
    tokens = torch.arange(batch_size * 4).reshape(batch_size, 4) % 10 + 3
    labels = (tokens + 1) % 15
    full_batch = {"input_ids": tokens, "labels": labels}
    batches, start = [], 0
    for size in sizes:
        batches.append({key: tensor[start:start + size] for key, tensor in full_batch.items()})
        start += size
    return full_batch, batches


@pytest.mark.parametrize("sizes", [(2, 2), (2, 2, 1), (1,)])
@pytest.mark.parametrize("max_norm", [None, 0.05, 1.0])
def test_sft_accumulation_matches_one_full_batch_update(tiny_train_model, sizes, max_norm):
    full_batch, batches = make_batches(sizes)
    reference = deepcopy(tiny_train_model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
    expected_loss = compute_sft_loss(reference, full_batch)
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm if max_norm is not None else math.inf)
    reference_optimizer.step()

    optimizer = torch.optim.AdamW(tiny_train_model.parameters(), lr=0.01, weight_decay=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    tiny_train_model.eval()
    for parameter in tiny_train_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    loss, metadata = sft_train_step(tiny_train_model, optimizer, batches, max_norm)

    optimizer.step.assert_called_once()
    assert tiny_train_model.training
    assert not loss.requires_grad
    assert not metadata["grad_norm"].requires_grad
    assert metadata["num_tokens"] == full_batch["labels"].numel()
    assert metadata["num_microbatches"] == len(sizes)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(metadata["grad_norm"], expected_norm, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.grad is None


def test_sft_gradient_clipping_happens_once_after_all_backwards(tiny_train_model, monkeypatch):
    _, batches = make_batches((2, 1))
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    observed_norms = []

    def inspect_gradients():
        norm = torch.stack([parameter.grad.norm() ** 2 for parameter in tiny_train_model.parameters() if parameter.grad is not None]).sum().sqrt()
        observed_norms.append(norm.item())

    optimizer.step = Mock(side_effect=inspect_gradients)
    clip = Mock(wraps=torch.nn.utils.clip_grad_norm_)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    _, metadata = sft_train_step(tiny_train_model, optimizer, batches, max_grad_norm=0.01)
    clip.assert_called_once()
    assert metadata["grad_norm"] > 0.01
    assert observed_norms[0] <= 0.010001


@pytest.mark.parametrize("kind", ["empty_group", "empty_batch", "invalid_shape", "negative_norm", "nan_norm", "inf_norm"])
def test_invalid_sft_group_never_updates_parameters(tiny_train_model, kind):
    _, batches = make_batches((2, 1))
    max_norm = 1.0
    if kind == "empty_group":
        batches = []
    elif kind == "empty_batch":
        batches[-1] = {"input_ids": torch.empty((0, 4), dtype=torch.long), "labels": torch.empty((0, 4), dtype=torch.long)}
    elif kind == "invalid_shape":
        batches[-1]["labels"] = batches[-1]["labels"][:, :-1]
    else:
        max_norm = {"negative_norm": -1.0, "nan_norm": float("nan"), "inf_norm": float("inf")}[kind]
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    initial = [parameter.detach().clone() for parameter in tiny_train_model.parameters()]
    with pytest.raises(ValueError):
        sft_train_step(tiny_train_model, optimizer, batches, max_norm)
    optimizer.step.assert_not_called()
    for parameter, before in zip(tiny_train_model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter, before)
        assert parameter.grad is None


def test_sft_nonfinite_gradients_abort_update_and_clear_accumulation(tiny_train_model, monkeypatch):
    _, batches = make_batches((2, 1))
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    parameter = next(tiny_train_model.parameters())
    monkeypatch.setattr(sft, "compute_sft_loss", lambda model, batch: parameter.sum() * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        sft_train_step(tiny_train_model, optimizer, batches)
    optimizer.step.assert_not_called()
    assert all(parameter.grad is None for parameter in tiny_train_model.parameters())


def test_sft_step_accepts_a_smaller_last_dataloader_batch(tiny_train_model, tokenizer):
    dataset = PackedSFTDataset(tokenizer, FIXTURES_PATH / "sft_sample.jsonl", seq_length=8, shuffle=False)
    batch_size = 7 if len(dataset) % 7 else 11
    batches = list(iterate_batches(dataset, batch_size=batch_size, shuffle=False))[-2:]
    assert batches[-1]["labels"].shape[0] < batch_size
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, metadata = sft_train_step(tiny_train_model, optimizer, batches)
    optimizer.step.assert_called_once()
    assert torch.isfinite(loss)
    assert metadata["num_tokens"] == sum(batch["labels"].numel() for batch in batches)
