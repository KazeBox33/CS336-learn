import pytest
import torch

from cs336_alignment.policy import compute_policy_gradient_loss, get_old_log_probs
from cs336_alignment.tokenization import tokenize_prompt_and_output
from cs336_alignment.training import train_rollout_batch


def test_grpo_counts_only_active_response_tokens_and_directional_clipping():
    ratios = torch.tensor([[1.5, 0.5, 1.0, 9.0], [1.5, 0.5, 1.0, 9.0], [9.0] * 4, [0.5] * 4])
    log_probs = ratios.log().requires_grad_()
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0, 0.0, 1.0])
    loss, metadata = compute_policy_gradient_loss(advantages, log_probs, "grpo", torch.zeros_like(log_probs), 0.2, mask)
    assert metadata["clip_unit_count"] == 6
    assert metadata["clipped_unit_count"] == 2
    assert metadata["clip_fraction"].item() == pytest.approx(1 / 3)
    assert all(not value.requires_grad for value in metadata.values())
    loss.masked_fill(~mask, 0).sum().backward()
    assert log_probs.grad[0, 0] == log_probs.grad[1, 1] == 0
    assert log_probs.grad[0, 1] < 0
    assert log_probs.grad[1, 0] > 0


def test_gspo_counts_responses_not_broadcast_token_weights():
    log_probs = torch.tensor([[1.5] * 4, [0.5] * 4, [9.0] * 4, [9.0] * 4]).log().requires_grad_()
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    _, metadata = compute_policy_gradient_loss(
        torch.tensor([1.0, -1.0, 0.0, 1.0]), log_probs, "gspo", torch.zeros_like(log_probs), 0.2, mask,
    )
    assert metadata["clip_unit_count"] == 2
    assert metadata["clipped_unit_count"] == 2
    assert metadata["clip_fraction"] == 1


@pytest.mark.parametrize("method", ["grpo", "gspo"])
@pytest.mark.parametrize("train_size, accumulation", [(4, 1), (4, 3), (2, 1), (2, 2)])
def test_clip_counts_are_weighted_across_microbatches_and_updates(
    tiny_train_model, tokenizer, method, train_size, accumulation,
):
    batch = tokenize_prompt_and_output(["Hello"] * 4, ["world", "a test", "another test", "test"], tokenizer)
    advantages = torch.tensor([1.0, -1.0, 0.0, 1.0])
    current = get_old_log_probs(tiny_train_model, batch["input_ids"], batch["labels"], microbatch_size=2)
    ratios = torch.tensor([1.5, 1.5, 9.0, 0.5]).unsqueeze(-1)
    old = current - ratios.log()
    expected_clipped = batch["response_mask"][0].sum() if method == "grpo" else 1
    expected_total = batch["response_mask"][advantages != 0].sum() if method == "grpo" else 3
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.0)
    _, metadata = train_rollout_batch(
        tiny_train_model, optimizer, batch, advantages, train_size, accumulation, 1.0,
        importance_reweighting_method=method, old_log_probs=old, cliprange=0.2,
    )
    assert metadata["clipped_unit_count"] == expected_clipped
    assert metadata["clip_unit_count"] == expected_total
    assert metadata["clip_fraction"].item() == pytest.approx(float(expected_clipped / expected_total))


@pytest.mark.parametrize("method", ["grpo", "gspo"])
def test_all_zero_advantages_report_zero_count_without_nan(tiny_train_model, tokenizer, method):
    batch = tokenize_prompt_and_output(["Hello"] * 2, ["world", "test"], tokenizer)
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.0)
    _, metadata = train_rollout_batch(
        tiny_train_model, optimizer, batch, torch.zeros(2), 1, 1, 1.0,
        importance_reweighting_method=method, old_log_probs=torch.zeros_like(batch["labels"]), cliprange=0.2,
    )
    assert metadata["clip_unit_count"] == metadata["clipped_unit_count"] == metadata["clip_fraction"] == 0
