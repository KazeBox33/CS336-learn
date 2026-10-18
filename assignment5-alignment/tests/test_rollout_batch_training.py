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
from cs336_alignment.rewards import compute_group_normalized_rewards
from cs336_alignment.tokenization import tokenize_prompt_and_output
from cs336_alignment.training import _prepare_rollout_batch, train_rollout_batch


@pytest.fixture
def prepared_rollouts(tokenizer):
    def reward_fn(response, ground_truth):
        reward = {"world": 1.0, "test": 0.0, "a test": 0.5, "another test": 0.0}[response]
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    batch, advantages, _ = _prepare_rollout_batch(
        tokenizer, reward_fn, ["Hello"] * 6,
        ["world", "test", "a test", "another test", "world", "test"], ["42"] * 6, group_size=3,
    )
    return batch, advantages


@pytest.mark.parametrize("method", ["none", "noclip", "grpo", "gspo"])
@pytest.mark.parametrize("accumulation_steps", [1, 2])
def test_minibatches_match_sequential_updates_using_fixed_full_group_advantages(
    tiny_train_model, prepared_rollouts, method, accumulation_steps
):
    batch, advantages = prepared_rollouts
    advantages = advantages.requires_grad_()
    saved_advantages = advantages.detach().clone()
    reference = deepcopy(tiny_train_model)
    old_scores = get_old_log_probs(reference, batch["input_ids"], batch["labels"], microbatch_size=2)
    old_scores = (old_scores + torch.linspace(-0.4, 0.4, old_scores.numel()).reshape_as(old_scores)).requires_grad_()
    saved_old_scores = old_scores.detach().clone()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    losses, norms = [], []
    entropy_sum = torch.tensor(0.0)
    token_count = torch.tensor(0)
    for start in range(0, 6, 2):
        stop = start + 2
        scores = get_response_log_probs(reference, batch["input_ids"][start:stop], batch["labels"][start:stop], True)
        mask = batch["response_mask"][start:stop]
        mini_advantages = advantages[start:stop]
        token_losses, _ = compute_policy_gradient_loss(
            mini_advantages, scores["log_probs"], method, old_scores[start:stop], 0.2, mask
        )
        loss = aggregate_loss_across_microbatch(token_losses, mask)
        losses.append(loss.detach())
        active_tokens = mask & (mini_advantages.detach() != 0).unsqueeze(-1)
        entropy_sum += scores["token_entropy"].detach()[active_tokens].sum()
        token_count += active_tokens.sum()
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1))
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, metadata = train_rollout_batch(
        tiny_train_model, optimizer, batch, advantages, train_batch_size=2,
        gradient_accumulation_steps=accumulation_steps, max_grad_norm=0.1,
        importance_reweighting_method=method, old_log_probs=old_scores, cliprange=0.2,
    )
    assert optimizer.step.call_count == metadata["optimizer_steps"] == 3
    torch.testing.assert_close(loss, torch.stack(losses).mean(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["grad_norm"], torch.stack(norms).mean(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["grad_norm_max"], torch.stack(norms).max(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metadata["mean_token_entropy"], entropy_sum / token_count, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(old_scores, saved_old_scores)
    torch.testing.assert_close(advantages, saved_advantages)
    assert old_scores.grad is None
    assert advantages.grad is None
    assert not loss.requires_grad
    for actual, expected in zip(tiny_train_model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert actual.grad is None


def test_handout_256_responses_make_32_independent_updates(tiny_train_model, tokenizer):
    batch = tokenize_prompt_and_output(["Hello"] * 256, ["world", "test"] * 128, tokenizer)
    advantages, _ = compute_group_normalized_rewards(torch.tensor([1.0, 0.0] * 128), group_size=8)
    old_scores = get_old_log_probs(tiny_train_model, batch["input_ids"], batch["labels"], microbatch_size=8)
    saved_old_scores = old_scores.clone()
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, metadata = train_rollout_batch(
        tiny_train_model, optimizer, batch, advantages, train_batch_size=8,
        gradient_accumulation_steps=1, max_grad_norm=1.0,
        importance_reweighting_method="gspo", old_log_probs=old_scores, cliprange=3e-4,
    )
    assert optimizer.step.call_count == metadata["optimizer_steps"] == 32
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metadata.values() if isinstance(value, torch.Tensor))
    torch.testing.assert_close(old_scores, saved_old_scores)


def test_zero_advantage_minibatches_still_take_optimizer_steps(tiny_train_model, prepared_rollouts):
    batch, advantages = prepared_rollouts
    tiny_train_model.forward = Mock(wraps=tiny_train_model.forward)
    optimizer = torch.optim.AdamW(tiny_train_model.parameters(), lr=0.01, weight_decay=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    loss, metadata = train_rollout_batch(
        tiny_train_model, optimizer, batch, torch.zeros_like(advantages), train_batch_size=2,
        gradient_accumulation_steps=1, max_grad_norm=1.0,
        importance_reweighting_method="gspo", old_log_probs=torch.zeros_like(batch["labels"]), cliprange=3e-4,
    )
    tiny_train_model.forward.assert_not_called()
    assert optimizer.step.call_count == metadata["optimizer_steps"] == 3
    assert loss.item() == metadata["grad_norm"].item() == metadata["mean_token_entropy"].item() == 0.0


@pytest.mark.parametrize("train_size, accumulation_steps, bad_old", [
    (0, 1, False), (4, 1, False), (8, 1, False), (2, 0, False), (2, 3, False), (2, 1, True),
])
def test_invalid_batch_settings_fail_before_any_update(
    tiny_train_model, prepared_rollouts, train_size, accumulation_steps, bad_old
):
    batch, advantages = prepared_rollouts
    old_scores = torch.zeros_like(batch["labels"])
    if bad_old:
        old_scores = old_scores[:2]
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    with pytest.raises(ValueError):
        train_rollout_batch(
            tiny_train_model, optimizer, batch, advantages, train_size, accumulation_steps, 1.0,
            importance_reweighting_method="grpo", old_log_probs=old_scores, cliprange=0.2,
        )
    optimizer.step.assert_not_called()
