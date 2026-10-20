from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F

from cs336_alignment.supplement.data import PackedSFTDataset, iterate_batches
from cs336_alignment.supplement.sft import compute_sft_loss
from tests.common import FIXTURES_PATH


def test_sft_loss_matches_already_shifted_token_targets_and_gradients(tiny_train_model):
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]]),
        "labels": torch.tensor([[4, 5, 6, 1], [8, 9, 10, 1]]),
    }
    reference = deepcopy(tiny_train_model)
    logits = reference(input_ids=batch["input_ids"]).logits
    token_losses = -F.log_softmax(logits.float(), dim=-1).gather(-1, batch["labels"].unsqueeze(-1)).squeeze(-1)
    expected = token_losses.mean()
    expected.backward()

    original = [parameter.detach().clone() for parameter in tiny_train_model.parameters()]
    tiny_train_model.forward = Mock(wraps=tiny_train_model.forward)
    actual = compute_sft_loss(tiny_train_model, batch)
    assert actual.shape == torch.Size([])
    assert actual.requires_grad
    torch.testing.assert_close(actual, expected)
    assert tiny_train_model.forward.call_args.kwargs.keys() == {"input_ids"}
    actual.backward()
    for parameter, reference_parameter, initial in zip(tiny_train_model.parameters(), reference.parameters(), original, strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
        torch.testing.assert_close(parameter, initial)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sft_loss_uses_fp32_cross_entropy_and_accepts_noncontiguous_logits(dtype):
    class LogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor(
                [[[1000.0, -1000.0], [0.0, 0.0], [-1000.0, 1000.0]]], dtype=dtype,
            ).transpose(1, 2))

        def forward(self, input_ids):
            assert input_ids.shape == (1, 2)
            return SimpleNamespace(logits=self.scores)

    model = LogitModel()
    assert not model.scores.is_contiguous()
    batch = {"input_ids": torch.tensor([[0, 1]]), "labels": torch.tensor([[2, 0]])}
    loss = compute_sft_loss(model, batch)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor(2000.0))
    loss.backward()
    assert model.scores.grad is not None
    assert torch.isfinite(model.scores.grad).all()


def test_single_token_batch_is_not_shifted_or_dropped_again(tiny_train_model):
    batch = {"input_ids": torch.tensor([[3]]), "labels": torch.tensor([[4]])}
    expected = F.cross_entropy(tiny_train_model(input_ids=batch["input_ids"]).logits[:, 0].float(), torch.tensor([4]))
    torch.testing.assert_close(compute_sft_loss(tiny_train_model, batch), expected)


@pytest.mark.parametrize("input_shape, label_shape", [
    ((4,), (4,)), ((2, 4), (2, 3)), ((2, 4), (8,)), ((0, 4), (0, 4)), ((2, 0), (2, 0)),
])
def test_invalid_sft_batch_fails_before_forward(tiny_train_model, input_shape, label_shape):
    tiny_train_model.forward = Mock(wraps=tiny_train_model.forward)
    batch = {
        "input_ids": torch.zeros(input_shape, dtype=torch.long),
        "labels": torch.zeros(label_shape, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="non-empty"):
        compute_sft_loss(tiny_train_model, batch)
    tiny_train_model.forward.assert_not_called()


def test_packed_sft_dataloader_connects_to_real_cpu_model(tiny_train_model, tokenizer):
    dataset = PackedSFTDataset(tokenizer, FIXTURES_PATH / "sft_sample.jsonl", seq_length=8, shuffle=False)
    batch = next(iter(iterate_batches(dataset, batch_size=2, shuffle=False)))
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 8)
    loss = compute_sft_loss(tiny_train_model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in tiny_train_model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
