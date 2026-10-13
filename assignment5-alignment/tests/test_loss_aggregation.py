import pytest
import torch

from cs336_alignment.policy import (
    aggregate_loss_across_microbatch,
    compute_policy_gradient_loss,
    get_response_log_probs,
)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_averages_each_response_before_averaging_the_batch(mask_dtype):
    losses = torch.tensor([[99.0, 2.0, 4.0, 99.0], [8.0, 99.0, 99.0, 99.0]])
    mask = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=mask_dtype)
    result = aggregate_loss_across_microbatch(losses, mask)
    torch.testing.assert_close(result, torch.tensor(5.5))
    assert result.shape == ()


def test_gradient_is_zero_at_excluded_positions_and_length_scaled_elsewhere():
    losses = torch.zeros((2, 3), requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, True, False]])
    result = aggregate_loss_across_microbatch(losses, mask)
    result.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[0.25, 0.25, 0.0], [0.0, 0.5, 0.0]]))


def test_masked_nonfinite_values_do_not_contaminate_the_result():
    losses = torch.tensor([[float("nan"), 2.0, float("inf")]], requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    result = aggregate_loss_across_microbatch(losses, mask)
    torch.testing.assert_close(result, torch.tensor(2.0))
    result.backward()
    torch.testing.assert_close(losses.grad, torch.tensor([[0.0, 1.0, 0.0]]))


def test_empty_response_contributes_zero_to_the_batch_mean():
    losses = torch.tensor([[4.0, 99.0], [99.0, 99.0]])
    mask = torch.tensor([[True, False], [False, False]])
    torch.testing.assert_close(aggregate_loss_across_microbatch(losses, mask), torch.tensor(2.0))


def test_all_empty_responses_produce_a_differentiable_zero():
    losses = torch.ones((2, 3), requires_grad=True)
    result = aggregate_loss_across_microbatch(losses, torch.zeros_like(losses, dtype=torch.bool))
    torch.testing.assert_close(result, torch.tensor(0.0))
    result.backward()
    torch.testing.assert_close(losses.grad, torch.zeros_like(losses))


def test_aggregated_policy_loss_backpropagates_to_the_model(tiny_train_model):
    input_ids = torch.tensor([[3, 4, 5], [3, 5, 4]])
    labels = torch.tensor([[4, 5, 1], [5, 4, 1]])
    log_probs = get_response_log_probs(tiny_train_model, input_ids, labels)["log_probs"]
    token_losses, _ = compute_policy_gradient_loss(torch.tensor([1.0, -1.0]), log_probs)
    mask = torch.tensor([[False, True, True], [False, False, True]])
    loss = aggregate_loss_across_microbatch(token_losses, mask)
    loss.backward()
    gradients = [p.grad for p in tiny_train_model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize(
    "loss_shape, mask_shape",
    [((2,), (2,)), ((1, 2), (1, 3)), ((0, 2), (0, 2))],
)
def test_rejects_invalid_shapes_or_an_empty_batch(loss_shape, mask_shape):
    with pytest.raises(ValueError):
        aggregate_loss_across_microbatch(torch.zeros(loss_shape), torch.zeros(mask_shape))


def test_constant_normalization_is_reserved_for_a_later_step():
    with pytest.raises(NotImplementedError):
        aggregate_loss_across_microbatch(
            torch.ones((1, 2)), torch.ones((1, 2)), loss_normalization="constant", normalization_constant=42
        )
