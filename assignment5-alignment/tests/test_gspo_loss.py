import math

import pytest
import torch

from cs336_alignment.policy import aggregate_loss_across_microbatch, compute_policy_gradient_loss


@pytest.mark.parametrize("column_weights", [False, True])
def test_gspo_shares_response_weight_and_masks_values_and_gradients(column_weights):
    weights = torch.tensor([[1.0], [-1.0]] if column_weights else [1.0, -1.0], requires_grad=True)
    old_scores = torch.full((2, 3), -2.0, requires_grad=True)
    differences = torch.tensor([
        [math.log(1.1), float("nan"), math.log(1.1)],
        [math.log(0.9), math.log(0.9), float("inf")],
    ])
    current_scores = (old_scores.detach() + differences).requires_grad_()
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.bool)

    losses, _ = compute_policy_gradient_loss(weights, current_scores, "gspo", old_scores, 0.2, mask)
    assert losses.shape == current_scores.shape
    torch.testing.assert_close(losses, torch.tensor([[-1.1] * 3, [0.9] * 3]))
    aggregate_loss_across_microbatch(losses, mask).backward()
    torch.testing.assert_close(current_scores.grad, torch.tensor([[-0.275, 0.0, -0.275], [0.225, 0.225, 0.0]]))
    assert weights.grad is None
    assert old_scores.grad is None


def test_gspo_clips_shared_sequence_ratio_not_individual_token_ratios():
    weights = torch.tensor([1.0, -1.0, 1.0, -1.0])
    old_scores = torch.full((4, 2), -2.0)
    ratios = torch.tensor([[2.0, 0.8], [0.5, 0.5], [0.25, 1.0], [4.0, 1.0]])
    current_scores = (old_scores + ratios.log()).requires_grad_()
    mask = torch.ones_like(current_scores, dtype=torch.bool)

    losses, _ = compute_policy_gradient_loss(weights, current_scores, "gspo", old_scores, 0.2, mask)
    torch.testing.assert_close(losses, torch.tensor([[-1.2] * 2, [0.8] * 2, [-0.5] * 2, [2.0] * 2]))
    aggregate_loss_across_microbatch(losses, mask).backward()
    torch.testing.assert_close(current_scores.grad, torch.tensor([[0.0] * 2, [0.0] * 2, [-0.0625] * 2, [0.25] * 2]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gspo_averages_log_ratios_before_exp_to_avoid_intermediate_overflow(dtype):
    current_scores = torch.tensor([[-10.0, -170.0]], dtype=dtype, requires_grad=True)
    old_scores = torch.tensor([[-170.0, -10.0]], dtype=dtype)
    mask = torch.ones_like(current_scores, dtype=torch.bool)
    losses, _ = compute_policy_gradient_loss(torch.ones(1), current_scores, "gspo", old_scores, 0.2, mask)
    torch.testing.assert_close(losses, -torch.ones_like(losses))
    aggregate_loss_across_microbatch(losses, mask).backward()
    torch.testing.assert_close(current_scores.grad, torch.full_like(current_scores, -0.5))


def test_empty_response_contributes_zero_after_masked_aggregation():
    current_scores = torch.full((1, 3), float("nan"), requires_grad=True)
    mask = torch.zeros_like(current_scores, dtype=torch.bool)
    losses, _ = compute_policy_gradient_loss(torch.ones(1), current_scores, "gspo", torch.zeros((1, 3)), 0.2, mask)
    loss = aggregate_loss_across_microbatch(losses, mask)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(current_scores.grad, torch.zeros_like(current_scores))


@pytest.mark.parametrize("mask", [None, torch.ones(3), torch.ones((1, 3))])
def test_gspo_requires_response_mask_aligned_with_scores(mask):
    with pytest.raises(ValueError, match="response_mask"):
        compute_policy_gradient_loss(torch.ones(2), torch.zeros((2, 3)), "gspo", torch.zeros((2, 3)), 0.2, mask)


@pytest.mark.parametrize("cliprange", [None, -0.1, float("nan"), float("inf")])
def test_gspo_requires_valid_cliprange(cliprange):
    with pytest.raises(ValueError, match="cliprange"):
        compute_policy_gradient_loss(
            torch.ones(1), torch.zeros((1, 2)), "gspo", torch.zeros((1, 2)), cliprange, torch.ones((1, 2))
        )
