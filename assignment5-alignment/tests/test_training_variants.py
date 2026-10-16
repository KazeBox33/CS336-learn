from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_response_log_probs,
)
from cs336_alignment.rewards import compute_group_normalized_rewards
from cs336_alignment.tokenization import tokenize_prompt_and_output
from cs336_alignment.training import _accumulate_microbatch_gradients, grpo_train_step


@pytest.fixture
def variant_inputs():
    def reward_fn(response, ground_truth):
        reward = float(response != "another test")
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    return {
        "reward_fn": reward_fn,
        "repeated_prompts": ["Hello", "Hello", "This is", "This is"],
        "rollout_responses": ["world", "test", "a test", "another test"],
        "repeated_ground_truths": ["42"] * 4,
        "group_size": 2,
    }


@pytest.mark.parametrize("baseline", ["mean", "none"])
@pytest.mark.parametrize("normalizer", ["std", "none", "mean"])
@pytest.mark.parametrize("loss_normalization", ["sequence", "constant"])
@pytest.mark.parametrize("steps", [1, 3, 4])
def test_all_on_policy_combinations_match_unpruned_update(
    tiny_train_model, tokenizer, variant_inputs, baseline, normalizer, loss_normalization, steps
):
    batch = tokenize_prompt_and_output(
        variant_inputs["repeated_prompts"], variant_inputs["rollout_responses"], tokenizer
    )
    advantages, _ = compute_group_normalized_rewards(
        torch.tensor([1.0, 1.0, 1.0, 0.0]), 2,
        baseline=baseline, advantage_normalizer=normalizer,
    )
    reference = deepcopy(tiny_train_model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    scores = get_response_log_probs(reference, batch["input_ids"], batch["labels"], True)
    token_losses, _ = compute_policy_gradient_loss(advantages, scores["log_probs"])
    expected_loss = aggregate_loss_across_microbatch(
        token_losses, batch["response_mask"], loss_normalization, 32
    )
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
    reference_optimizer.step()
    active = advantages != 0
    active_token_mask = batch["response_mask"] & active.unsqueeze(-1)
    expected_entropy = scores["token_entropy"].detach()[active_token_mask].mean()

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    tiny_train_model.forward = Mock(wraps=tiny_train_model.forward)
    loss, metadata = grpo_train_step(
        tiny_train_model, tokenizer, optimizer, steps, 0.1, **variant_inputs,
        baseline=baseline, advantage_normalizer=normalizer,
        loss_normalization=loss_normalization, normalization_constant=32,
    )
    torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["grad_norm"], expected_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["mean_token_entropy"], expected_entropy)
    assert metadata["mean_reward"] == 0.75
    assert metadata["mean_format_reward"] == 1.0
    forwarded = torch.cat([
        call.kwargs["input_ids"] for call in tiny_train_model.forward.call_args_list
    ])
    torch.testing.assert_close(forwarded, batch["input_ids"][active])
    assert tiny_train_model.forward.call_count <= steps
    if steps > 1:
        assert tiny_train_model.forward.call_count < steps
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.grad is None


@pytest.mark.parametrize("loss_normalization", ["sequence", "constant"])
@pytest.mark.parametrize("steps", [1, 2, 3, 5])
def test_pruning_preserves_gradients_with_uneven_microbatches(
    tiny_train_model, tokenizer, loss_normalization, steps
):
    batch = tokenize_prompt_and_output(
        ["Hello"] * 5, ["world", "test", "a test", "another test", "world"], tokenizer
    )
    advantages = torch.tensor([0.0, 1.0, -0.5, 0.0, 0.25])
    reference = deepcopy(tiny_train_model)
    scores = get_response_log_probs(reference, batch["input_ids"], batch["labels"])
    token_losses, _ = compute_policy_gradient_loss(advantages, scores["log_probs"])
    expected_loss = aggregate_loss_across_microbatch(
        token_losses, batch["response_mask"], loss_normalization, 37
    )
    expected_loss.backward()
    loss, _ = _accumulate_microbatch_gradients(
        tiny_train_model, batch, advantages, steps, loss_normalization, 37
    )
    torch.testing.assert_close(loss, expected_loss.detach(), rtol=1e-5, atol=1e-6)
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("baseline, reward", [("mean", 1.0), ("none", 0.0)])
@pytest.mark.parametrize("normalizer", ["std", "none", "mean"])
@pytest.mark.parametrize("loss_normalization", ["sequence", "constant"])
def test_all_zero_advantages_skip_forward_but_preserve_adamw_update(
    tiny_train_model, tokenizer, variant_inputs, baseline, reward, normalizer, loss_normalization
):
    reference = deepcopy(tiny_train_model)
    optimizer = torch.optim.AdamW(tiny_train_model.parameters(), lr=0.01, weight_decay=0.1)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
    for model, opt in [(tiny_train_model, optimizer), (reference, reference_optimizer)]:
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
        opt.step()
        opt.zero_grad(set_to_none=True)

    batch = tokenize_prompt_and_output(
        variant_inputs["repeated_prompts"], variant_inputs["rollout_responses"], tokenizer
    )
    scores = get_response_log_probs(reference, batch["input_ids"], batch["labels"])
    token_losses, _ = compute_policy_gradient_loss(torch.zeros(4), scores["log_probs"])
    aggregate_loss_across_microbatch(
        token_losses, batch["response_mask"], loss_normalization, 32
    ).backward()
    reference_optimizer.step()

    def reward_fn(response, ground_truth):
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    inputs = {**variant_inputs, "reward_fn": reward_fn}
    tiny_train_model.forward = Mock(wraps=tiny_train_model.forward)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, metadata = grpo_train_step(
        tiny_train_model, tokenizer, optimizer, 3, None, **inputs,
        baseline=baseline, advantage_normalizer=normalizer,
        loss_normalization=loss_normalization, normalization_constant=32,
    )
    tiny_train_model.forward.assert_not_called()
    optimizer.step.assert_called_once()
    assert loss.item() == 0.0
    assert metadata["grad_norm"].item() == 0.0
    assert metadata["mean_token_entropy"].item() == 0.0
    assert metadata["mean_reward"] == reward
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[actual][key], reference_optimizer.state[expected][key])
