from copy import deepcopy

import pytest
import torch

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_response_log_probs,
)
from cs336_alignment.tokenization import tokenize_prompt_and_output
from cs336_alignment.training import _accumulate_microbatch_gradients


@pytest.fixture
def rollout_tensors(tokenizer):
    batch = tokenize_prompt_and_output(
        ["Hello"] * 5, ["world", "test", "a test", "another test", "world"], tokenizer
    )
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0])
    return batch, advantages


def _whole_batch_backward(model, batch, advantages):
    scores = get_response_log_probs(model, batch["input_ids"], batch["labels"])
    token_losses, _ = compute_policy_gradient_loss(advantages, scores["log_probs"])
    loss = aggregate_loss_across_microbatch(token_losses, batch["response_mask"])
    loss.backward()
    return loss.detach()


@pytest.mark.parametrize("steps", [1, 2, 3, 5])
def test_matches_full_batch_loss_and_gradients_without_updating_parameters(
    tiny_train_model, rollout_tensors, steps
):
    batch, advantages = rollout_tensors
    reference_model = deepcopy(tiny_train_model)
    expected_loss = _whole_batch_backward(reference_model, batch, advantages)
    original_parameters = [p.detach().clone() for p in tiny_train_model.parameters()]
    loss, metadata = _accumulate_microbatch_gradients(tiny_train_model, batch, advantages, steps)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
    assert loss.shape == ()
    assert not loss.requires_grad
    assert torch.isfinite(metadata["mean_token_entropy"])
    assert not metadata["mean_token_entropy"].requires_grad
    for param, reference, original in zip(
        tiny_train_model.parameters(), reference_model.parameters(), original_parameters, strict=True
    ):
        torch.testing.assert_close(param, original)
        if reference.grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, reference.grad, rtol=1e-5, atol=1e-6)


def test_accumulates_into_existing_gradients_instead_of_clearing_them(tiny_train_model, rollout_tensors):
    batch, advantages = rollout_tensors
    reference_model = deepcopy(tiny_train_model)
    _whole_batch_backward(reference_model, batch, advantages)
    for param in tiny_train_model.parameters():
        param.grad = torch.full_like(param, 0.125)
    _accumulate_microbatch_gradients(tiny_train_model, batch, advantages, 2)
    for param, reference in zip(tiny_train_model.parameters(), reference_model.parameters(), strict=True):
        expected = torch.full_like(param, 0.125)
        if reference.grad is not None:
            expected += reference.grad
        torch.testing.assert_close(param.grad, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("steps", [0, 6])
def test_rejects_invalid_microbatch_counts(tiny_train_model, rollout_tensors, steps):
    batch, advantages = rollout_tensors
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        _accumulate_microbatch_gradients(tiny_train_model, batch, advantages, steps)
    assert all(p.grad is None for p in tiny_train_model.parameters())


@pytest.mark.parametrize("advantages", [torch.empty(0), torch.ones((5, 1)), torch.ones(4)])
def test_rejects_invalid_advantages_or_unaligned_token_batches(tiny_train_model, rollout_tensors, advantages):
    batch, _ = rollout_tensors
    with pytest.raises(ValueError):
        _accumulate_microbatch_gradients(tiny_train_model, batch, advantages, 1)
    assert all(p.grad is None for p in tiny_train_model.parameters())
