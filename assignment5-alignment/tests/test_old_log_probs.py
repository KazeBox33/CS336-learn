import pytest
import torch

from cs336_alignment import policy


@pytest.mark.parametrize("microbatch_size", [1, 2, 4])
@pytest.mark.parametrize("training", [False, True])
def test_old_scores_match_full_batch_without_changing_model(tiny_train_model, microbatch_size, training):
    model = tiny_train_model
    input_ids = torch.tensor([[3, 4, 5], [3, 5, 4], [5, 4, 3]])
    labels = torch.tensor([[4, 5, 1], [5, 4, 1], [4, 3, 1]])
    model.eval()
    with torch.no_grad():
        expected = policy.get_response_log_probs(model, input_ids, labels)["log_probs"]
    model.train(training)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients_before = [parameter.grad for parameter in model.parameters()]

    old_log_probs = policy.get_old_log_probs(model, input_ids, labels, microbatch_size)

    torch.testing.assert_close(old_log_probs, expected)
    assert old_log_probs.shape == labels.shape
    assert old_log_probs.device.type == "cpu"
    assert not old_log_probs.requires_grad
    assert old_log_probs.grad_fn is None
    assert model.training == training
    for parameter, before, gradient in zip(model.parameters(), parameters_before, gradients_before):
        torch.testing.assert_close(parameter, before)
        assert parameter.grad is gradient
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


def test_cached_scores_stay_fixed_after_optimizer_update(tiny_train_model):
    model = tiny_train_model
    input_ids = torch.tensor([[3, 4, 5]])
    labels = torch.tensor([[4, 5, 1]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    old_log_probs = policy.get_old_log_probs(model, input_ids, labels, microbatch_size=1)
    saved_scores = old_log_probs.clone()
    current_scores = policy.get_response_log_probs(model, input_ids, labels)["log_probs"]
    (-current_scores.mean()).backward()
    optimizer.step()
    new_scores = policy.get_response_log_probs(model, input_ids, labels)["log_probs"]

    torch.testing.assert_close(old_log_probs, saved_scores)
    assert not torch.allclose(new_scores, old_log_probs)


def test_scoring_disables_grad_and_dropout_and_restores_mode_on_error(tiny_train_model, monkeypatch):
    def fail(model, input_ids, labels):
        assert not model.training
        assert not torch.is_grad_enabled()
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(policy, "get_response_log_probs", fail)
    with pytest.raises(RuntimeError, match="scoring failed"):
        policy.get_old_log_probs(tiny_train_model, torch.ones((1, 2), dtype=torch.long),
                                 torch.ones((1, 2), dtype=torch.long), microbatch_size=1)
    assert tiny_train_model.training


@pytest.mark.parametrize(
    "input_ids, labels, microbatch_size",
    [
        (torch.ones(2), torch.ones(2), 1),
        (torch.ones((1, 2)), torch.ones((2, 2)), 1),
        (torch.empty((0, 2)), torch.empty((0, 2)), 1),
        (torch.ones((1, 2)), torch.ones((1, 2)), 0),
    ],
)
def test_old_scores_reject_invalid_batches(tiny_train_model, input_ids, labels, microbatch_size):
    with pytest.raises(ValueError):
        policy.get_old_log_probs(tiny_train_model, input_ids, labels, microbatch_size)
