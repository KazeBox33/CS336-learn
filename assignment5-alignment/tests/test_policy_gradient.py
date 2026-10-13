import pytest
import torch

from cs336_alignment.policy import compute_policy_gradient_loss, get_response_log_probs


@pytest.mark.parametrize("column_weights", [False, True])
def test_broadcasts_each_response_weight_over_its_tokens(column_weights):
    weights = torch.tensor([1.0, -1.0])
    if column_weights:
        weights = weights.unsqueeze(-1)
    log_probs = torch.tensor([[-0.2, -1.0], [-0.3, -0.5]])
    loss, metadata = compute_policy_gradient_loss(weights, log_probs)
    torch.testing.assert_close(loss, torch.tensor([[0.2, 1.0], [-0.3, -0.5]]))
    assert loss.shape == log_probs.shape
    assert metadata == {}


def test_zero_advantage_produces_zero_loss_and_gradient():
    log_probs = torch.tensor([[-0.2, -1.0]], requires_grad=True)
    loss, _ = compute_policy_gradient_loss(torch.zeros(1), log_probs)
    torch.testing.assert_close(loss, torch.zeros_like(log_probs))
    loss.sum().backward()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))


def test_gradient_flows_through_log_probs_not_advantages():
    weights = torch.tensor([2.0, -1.0], requires_grad=True)
    log_probs = torch.tensor([[-0.2, -1.0], [-0.3, -0.5]], requires_grad=True)
    loss, _ = compute_policy_gradient_loss(weights, log_probs)
    loss.sum().backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([[-2.0, -2.0], [1.0, 1.0]]))
    assert weights.grad is None


def test_loss_backpropagates_into_a_causal_model(tiny_train_model):
    input_ids = torch.tensor([[3, 4, 5], [3, 5, 4]])
    labels = torch.tensor([[4, 5, 1], [5, 4, 1]])
    log_probs = get_response_log_probs(tiny_train_model, input_ids, labels)["log_probs"]
    loss, _ = compute_policy_gradient_loss(torch.tensor([1.0, -1.0]), log_probs)
    loss.mean().backward()
    gradients = [p.grad for p in tiny_train_model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
def test_later_reweighting_methods_are_explicitly_unsupported(method):
    with pytest.raises(NotImplementedError):
        compute_policy_gradient_loss(
            torch.ones(1), torch.zeros((1, 2)), importance_reweighting_method=method
        )


@pytest.mark.parametrize(
    "weights, log_probs",
    [
        (torch.tensor(1.0), torch.zeros((1, 2))),
        (torch.ones(2), torch.zeros((1, 2))),
        (torch.ones((1, 2)), torch.zeros((1, 2))),
        (torch.ones(1), torch.zeros(2)),
    ],
)
def test_rejects_shapes_that_could_broadcast_incorrectly(weights, log_probs):
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(weights, log_probs)


def test_on_policy_ignores_parameters_reserved_for_later_steps():
    log_probs = torch.tensor([[-0.2, -1.0]])
    loss, _ = compute_policy_gradient_loss(
        torch.ones(1),
        log_probs,
        old_log_probs=torch.zeros_like(log_probs),
        cliprange=0.1,
        response_mask=torch.zeros_like(log_probs, dtype=torch.bool),
    )
    torch.testing.assert_close(loss, -log_probs)
