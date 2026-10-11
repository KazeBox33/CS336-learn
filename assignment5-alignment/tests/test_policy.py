from types import SimpleNamespace

import pytest
import torch

from cs336_alignment.policy import get_response_log_probs


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, input_ids: torch.Tensor):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


def test_selects_each_label_probability():
    model = FixedLogitsModel(torch.tensor([0.2, 0.3, 0.5]).log())
    input_ids = torch.zeros((2, 2), dtype=torch.long)
    labels = torch.tensor([[0, 2], [1, 0]])
    result = get_response_log_probs(model, input_ids, labels)
    expected = torch.tensor([[0.2, 0.5], [0.3, 0.2]]).log()
    torch.testing.assert_close(result["log_probs"], expected)
    assert set(result) == {"log_probs"}


def test_entropy_uses_the_entire_vocabulary_distribution():
    probabilities = torch.tensor([0.2, 0.3, 0.5])
    model = FixedLogitsModel(probabilities.log())
    input_ids = torch.zeros((1, 2), dtype=torch.long)
    labels = torch.tensor([[0, 2]])
    result = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    expected = -(probabilities * probabilities.log()).sum()
    torch.testing.assert_close(result["token_entropy"], expected.expand(1, 2))


def test_log_probs_retain_gradients_for_policy_training():
    model = FixedLogitsModel(torch.tensor([0.0, 1.0, 2.0]))
    input_ids = torch.zeros((1, 2), dtype=torch.long)
    labels = torch.tensor([[0, 1]])
    result = get_response_log_probs(model, input_ids, labels)
    loss = -result["log_probs"].mean()
    loss.backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()
    assert model.logits.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_logits_remain_finite(dtype):
    model = FixedLogitsModel(torch.tensor([10000.0, 0.0, -10000.0], dtype=dtype))
    input_ids = torch.zeros((1, 3), dtype=torch.long)
    labels = torch.tensor([[0, 1, 2]])
    result = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    for tensor in result.values():
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(result["token_entropy"], torch.zeros((1, 3)))


@pytest.mark.parametrize("shape", [(2,), (1, 3)])
def test_rejects_invalid_input_shapes(shape):
    model = FixedLogitsModel(torch.zeros(3))
    input_ids = torch.zeros(shape, dtype=torch.long)
    labels = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="two-dimensional shape"):
        get_response_log_probs(model, input_ids, labels)
