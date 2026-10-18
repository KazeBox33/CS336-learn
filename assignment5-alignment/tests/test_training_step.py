from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_response_log_probs,
)
from cs336_alignment.training import _prepare_rollout_batch, grpo_train_step


def _reward(response, ground_truth):
    rewards = {"world": 1.0, "test": 0.0, "a test": 0.5, "another test": 0.0}
    reward = rewards[response]
    return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}


@pytest.fixture
def step_inputs():
    return {
        "reward_fn": _reward,
        "repeated_prompts": ["Hello", "Hello", "This is", "This is"],
        "rollout_responses": ["world", "test", "a test", "another test"],
        "repeated_ground_truths": ["42"] * 4,
        "group_size": 2,
    }


@pytest.mark.parametrize("steps, max_norm", [(1, 0.05), (2, 0.05), (3, None)])
def test_update_matches_full_batch_and_logs_pre_update_statistics(
    tiny_train_model, tokenizer, step_inputs, steps, max_norm
):
    reference = deepcopy(tiny_train_model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    batch, advantages, _ = _prepare_rollout_batch(tokenizer=tokenizer, **step_inputs)
    scores = get_response_log_probs(reference, batch["input_ids"], batch["labels"], True)
    token_loss, _ = compute_policy_gradient_loss(advantages, scores["log_probs"])
    expected_loss = aggregate_loss_across_microbatch(token_loss, batch["response_mask"])
    expected_entropy = scores["token_entropy"].detach()[batch["response_mask"]].mean()
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(
        reference.parameters(), max_norm if max_norm is not None else float("inf")
    )
    reference_optimizer.step()

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    # The train step must discard stale gradients and enter training mode.
    tiny_train_model.eval()
    for param in tiny_train_model.parameters():
        param.grad = torch.full_like(param, 0.25)
    loss, metadata = grpo_train_step(
        model=tiny_train_model, tokenizer=tokenizer, optimizer=optimizer,
        gradient_accumulation_steps=steps, max_grad_norm=max_norm, **step_inputs,
    )
    optimizer.step.assert_called_once()
    assert tiny_train_model.training
    assert not loss.requires_grad
    torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["grad_norm"], expected_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["mean_token_entropy"], expected_entropy, rtol=1e-5, atol=1e-6)
    assert metadata["mean_reward"] == 0.375
    assert metadata["mean_format_reward"] == 1.0
    if max_norm is not None:
        assert metadata["grad_norm"] > max_norm
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.grad is None
    assert all(not value.requires_grad for value in metadata.values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"baseline": "invalid"}, ValueError),
        ({"advantage_normalizer": "invalid"}, NotImplementedError),
        ({"importance_reweighting_method": "gspo"}, NotImplementedError),
        ({"importance_reweighting_method": "grpo"}, ValueError),
        ({"loss_normalization": "constant"}, ValueError),
        ({"loss_normalization": "invalid"}, NotImplementedError),
        ({"loss_normalization": "constant", "normalization_constant": 0}, ValueError),
        ({"gradient_accumulation_steps": 0}, ValueError),
        ({"gradient_accumulation_steps": 5}, ValueError),
        ({"max_grad_norm": -1.0}, ValueError),
        ({"max_grad_norm": float("nan")}, ValueError),
        ({"advantage_eps": float("nan")}, ValueError),
    ],
)
def test_invalid_settings_do_not_update_parameters(tiny_train_model, tokenizer, step_inputs, kwargs, error):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    settings = {"gradient_accumulation_steps": 2, "max_grad_norm": 1.0, **kwargs}
    original = [p.detach().clone() for p in tiny_train_model.parameters()]
    with pytest.raises(error):
        grpo_train_step(model=tiny_train_model, tokenizer=tokenizer, optimizer=optimizer, **settings, **step_inputs)
    optimizer.step.assert_not_called()
    for param, initial in zip(tiny_train_model.parameters(), original, strict=True):
        torch.testing.assert_close(param, initial)
