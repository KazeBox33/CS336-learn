import math
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cs336_alignment.supplement import dpo_training
from cs336_alignment.supplement.dpo import compute_per_instance_dpo_loss
from cs336_alignment.supplement.dpo_training import dpo_train_step
from cs336_alignment.supplement.hh import HHPreferenceExample


@pytest.fixture
def models(tokenizer):
    torch.manual_seed(12)
    policy = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_ctx=128,
        n_embd=8, n_layer=1, n_head=2, n_inner=16,
        resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2, use_cache=False,
    ))
    reference = deepcopy(policy)
    with torch.no_grad():
        reference.transformer.wte.weight[3].add_(0.1)
    return policy, reference


@pytest.fixture
def examples():
    return [
        HHPreferenceExample("Question", "Hello", "world", "helpful-base.jsonl.gz"),
        HHPreferenceExample("Instruction", "This is a test", "Hello world", "helpful-online.jsonl.gz"),
        HHPreferenceExample("Question", "Answer", "", "harmless-base.jsonl.gz"),
    ]


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("max_norm", [None, 0.01])
def test_dpo_accumulation_matches_mean_loss_rmsprop_update(models, tokenizer, examples, count, max_norm):
    policy, reference = models
    examples = examples[:count]
    expected_policy = deepcopy(policy)
    expected_reference = deepcopy(reference).requires_grad_(False).eval()
    expected_optimizer = torch.optim.RMSprop(expected_policy.parameters(), lr=0.001, weight_decay=0.1)
    expected_loss = torch.stack([
        compute_per_instance_dpo_loss(
            expected_policy, expected_reference, tokenizer, 0.1,
            example.instruction, example.chosen, example.rejected,
        ) for example in examples
    ]).mean()
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(
        expected_policy.parameters(), max_norm if max_norm is not None else math.inf,
    )
    expected_optimizer.step()
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=0.001, weight_decay=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    policy.eval()
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    original_reference = [parameter.detach().clone() for parameter in reference.parameters()]

    loss, metadata = dpo_train_step(policy, reference, tokenizer, optimizer, examples, max_grad_norm=max_norm)

    optimizer.step.assert_called_once()
    assert loss.shape == torch.Size([])
    assert not loss.requires_grad and not metadata["grad_norm"].requires_grad
    assert metadata["num_examples"] == count
    assert policy.training and not reference.training
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(metadata["grad_norm"], expected_norm, rtol=1e-4, atol=1e-6)
    for actual, expected in zip(policy.parameters(), expected_policy.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        assert actual.grad is None
    for parameter, original in zip(reference.parameters(), original_reference, strict=True):
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_dpo_backwards_each_pair_before_scoring_next(models, tokenizer, examples, monkeypatch):
    policy, reference = models
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    observed_gradients = []
    original = compute_per_instance_dpo_loss

    def inspect(*args):
        observed_gradients.append(any(parameter.grad is not None for parameter in policy.parameters()))
        return original(*args)

    monkeypatch.setattr(dpo_training, "compute_per_instance_dpo_loss", inspect)
    dpo_train_step(policy, reference, tokenizer, optimizer, examples)
    assert observed_gradients == [False, True, True]


def test_dpo_clips_once_after_accumulation(models, tokenizer, examples, monkeypatch):
    policy, reference = models
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    observed_norms = []

    def inspect(opt, args, kwargs):
        observed_norms.append(torch.stack([
            parameter.grad.norm() ** 2 for parameter in policy.parameters() if parameter.grad is not None
        ]).sum().sqrt().item())

    optimizer.register_step_pre_hook(inspect)
    clip = Mock(wraps=torch.nn.utils.clip_grad_norm_)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    _, metadata = dpo_train_step(policy, reference, tokenizer, optimizer, examples, max_grad_norm=0.01)
    clip.assert_called_once()
    assert metadata["grad_norm"] > 0.01
    assert observed_norms[0] <= 0.010001


@pytest.mark.parametrize("kind", ["empty", "invalid_example", "negative_norm", "nan_norm", "inf_norm"])
def test_dpo_bad_group_does_not_update_policy(models, tokenizer, examples, kind):
    policy, reference = models
    max_norm = None
    if kind == "empty":
        examples = []
    elif kind == "invalid_example":
        examples[1] = HHPreferenceExample("Question", None, "world", "test")
    else:
        max_norm = {"negative_norm": -1.0, "nan_norm": float("nan"), "inf_norm": float("inf")}[kind]
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    with pytest.raises((ValueError, TypeError)):
        dpo_train_step(policy, reference, tokenizer, optimizer, examples, max_grad_norm=max_norm)
    optimizer.step.assert_not_called()
    for parameter, original in zip(policy.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        assert parameter.grad is None


@pytest.mark.parametrize("shared", ["same_model", "shared_parameter", "wrong_optimizer", "incomplete_optimizer"])
def test_dpo_rejects_reference_policy_parameter_mixups(models, tokenizer, examples, shared):
    policy, reference = models
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    if shared == "same_model":
        reference = policy
    elif shared == "shared_parameter":
        reference.transformer.wte.weight = policy.transformer.wte.weight
    elif shared == "wrong_optimizer":
        optimizer.add_param_group({"params": list(reference.parameters())})
    else:
        optimizer = torch.optim.RMSprop([next(policy.parameters())], lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    with pytest.raises(ValueError, match="Parameter|parameters"):
        dpo_train_step(policy, reference, tokenizer, optimizer, examples)
    optimizer.step.assert_not_called()
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_dpo_nonfinite_gradient_prevents_update(models, tokenizer, examples, monkeypatch):
    policy, reference = models
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    monkeypatch.setattr(dpo_training, "compute_per_instance_dpo_loss", lambda *args: next(policy.parameters()).sum() * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        dpo_train_step(policy, reference, tokenizer, optimizer, examples)
    optimizer.step.assert_not_called()
    assert all(parameter.grad is None for parameter in policy.parameters())
