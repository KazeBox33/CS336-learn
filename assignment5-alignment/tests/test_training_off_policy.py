from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_old_log_probs,
    get_response_log_probs,
)
from cs336_alignment.training import _prepare_rollout_batch, grpo_train_step


@pytest.fixture
def rollout_inputs():
    def reward_fn(response, ground_truth):
        reward = {"world": 1.0, "test": 0.0, "a test": 0.5, "another test": 0.0}[response]
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    return {
        "reward_fn": reward_fn,
        "repeated_prompts": ["Hello"] * 5,
        "rollout_responses": ["world", "test", "a test", "another test", "world"],
        "repeated_ground_truths": ["42"] * 5,
        "group_size": 5,
    }


@pytest.mark.parametrize("method, normalization", [
    ("noclip", "sequence"), ("grpo", "sequence"), ("gspo", "sequence"),
    ("noclip", "constant"), ("grpo", "constant"),
])
@pytest.mark.parametrize("steps", [1, 3, 5])
def test_off_policy_update_matches_full_batch_with_pruning(
    tiny_train_model, tokenizer, rollout_inputs, method, steps, normalization
):
    batch, advantages, _ = _prepare_rollout_batch(tokenizer=tokenizer, **rollout_inputs)
    reference = deepcopy(tiny_train_model)
    scores = get_response_log_probs(reference, batch["input_ids"], batch["labels"])
    ratios = torch.linspace(0.5, 1.5, scores["log_probs"].numel()).reshape_as(scores["log_probs"])
    old_scores = (scores["log_probs"].detach() - ratios.log()).requires_grad_()
    saved_old_scores = old_scores.detach().clone()
    token_losses, _ = compute_policy_gradient_loss(
        advantages, scores["log_probs"], method, old_scores, 0.2, batch["response_mask"]
    )
    expected_loss = aggregate_loss_across_microbatch(token_losses, batch["response_mask"], normalization, 37)
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    reference_optimizer.step()

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, metadata = grpo_train_step(
        tiny_train_model, tokenizer, optimizer, steps, 0.1, **rollout_inputs,
        importance_reweighting_method=method, old_log_probs=old_scores, cliprange=0.2,
        loss_normalization=normalization, normalization_constant=37,
    )
    optimizer.step.assert_called_once()
    torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["grad_norm"], expected_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(old_scores, saved_old_scores)
    assert old_scores.grad is None
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.grad is None


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
def test_cached_sampling_scores_can_be_reused_for_two_updates(
    tiny_train_model, tokenizer, rollout_inputs, method
):
    model = tiny_train_model
    batch, _, _ = _prepare_rollout_batch(tokenizer=tokenizer, **rollout_inputs)
    old_scores = get_old_log_probs(model, batch["input_ids"], batch["labels"], microbatch_size=2)
    saved_old_scores = old_scores.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        loss, _ = grpo_train_step(
            model, tokenizer, optimizer, 3, 1.0, **rollout_inputs,
            importance_reweighting_method=method, old_log_probs=old_scores, cliprange=0.2,
        )
        assert torch.isfinite(loss)
        torch.testing.assert_close(old_scores, saved_old_scores)


@pytest.mark.parametrize(
    "method, old_scores, cliprange",
    [
        ("noclip", None, None),
        ("noclip", torch.zeros((5, 2)), None),
        ("grpo", torch.zeros((5, 1)), None),
        ("grpo", torch.zeros((5, 1)), -0.1),
        ("grpo", torch.zeros((5, 1)), float("nan")),
        ("gspo", None, 0.2),
        ("gspo", torch.zeros((5, 2)), 0.2),
        ("gspo", torch.zeros((5, 1)), None),
        ("gspo", torch.zeros((5, 1)), -0.1),
        ("gspo", torch.zeros((5, 1)), float("nan")),
    ],
)
def test_invalid_off_policy_inputs_fail_even_when_all_advantages_are_zero(
    tiny_train_model, tokenizer, rollout_inputs, method, old_scores, cliprange
):
    inputs = {**rollout_inputs, "rollout_responses": ["world"] * 5}
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    before = [parameter.detach().clone() for parameter in tiny_train_model.parameters()]
    with pytest.raises(ValueError):
        grpo_train_step(
            tiny_train_model, tokenizer, optimizer, 3, 1.0, **inputs,
            importance_reweighting_method=method, old_log_probs=old_scores, cliprange=cliprange,
        )
    optimizer.step.assert_not_called()
    for parameter, saved in zip(tiny_train_model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, saved)
