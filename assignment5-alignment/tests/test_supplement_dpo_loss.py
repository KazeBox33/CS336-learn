import math
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from cs336_alignment.supplement import dpo
from cs336_alignment.supplement.dpo import compute_per_instance_dpo_loss


@pytest.fixture
def dpo_models(tokenizer):
    torch.manual_seed(11)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_ctx=128,
        n_embd=8, n_layer=1, n_head=2, n_inner=16,
        resid_pdrop=0.3, embd_pdrop=0.3, attn_pdrop=0.3, use_cache=False,
    ))
    reference = deepcopy(model)
    with torch.no_grad():
        reference.transformer.wte.weight[3].add_(0.1)
    return model, reference


def document_tokens(tokenizer, prompt, response):
    template = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )
    return tokenizer.encode(template, add_special_tokens=True) + [tokenizer.eos_token_id]


def negative_cross_entropy_score(model, ids):
    tokens = torch.tensor(ids, device=next(model.parameters()).device)
    logits = model(input_ids=tokens[:-1].unsqueeze(0)).logits.squeeze(0)
    return -F.cross_entropy(logits.float(), tokens[1:], reduction="sum")


def test_dpo_loss_and_policy_gradients_match_independent_sequence_calculation(dpo_models, tokenizer):
    model, reference = dpo_models
    expected_model = deepcopy(model).eval()
    expected_ref = deepcopy(reference).eval()
    chosen = document_tokens(tokenizer, "Question", "Hello world")
    rejected = document_tokens(tokenizer, "Question", "This is another test")
    expected_margin = negative_cross_entropy_score(expected_model, chosen) - negative_cross_entropy_score(expected_model, rejected)
    with torch.no_grad():
        expected_ref_margin = negative_cross_entropy_score(expected_ref, chosen) - negative_cross_entropy_score(expected_ref, rejected)
    expected = F.softplus(-0.1 * (expected_margin - expected_ref_margin))
    expected.backward()
    initial = [parameter.detach().clone() for parameter in model.parameters()]

    actual = compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello world", "This is another test")
    assert actual.shape == torch.Size([])
    assert actual.device == next(model.parameters()).device
    assert actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.backward()
    for parameter, target, before in zip(model.parameters(), expected_model.parameters(), initial, strict=True):
        torch.testing.assert_close(parameter.grad, target.grad, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert any(parameter.grad.abs().sum() > 0 for parameter in model.parameters() if parameter.grad is not None)
    assert all(parameter.grad is None and parameter.requires_grad for parameter in reference.parameters())


def test_dpo_scores_shifted_alpaca_documents_and_eos_with_correct_grad_modes(dpo_models, tokenizer, monkeypatch):
    model, reference = dpo_models
    calls = []
    original = dpo.get_response_log_probs

    def inspect(model_arg, input_ids, labels):
        calls.append((model_arg, input_ids.clone(), labels.clone(), torch.is_grad_enabled(), model_arg.training))
        return original(model_arg, input_ids, labels)

    monkeypatch.setattr(dpo, "get_response_log_probs", inspect)
    compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "Hello world")
    assert len(calls) == 4
    for scorer in (model, reference):
        model_calls = [call for call in calls if call[0] is scorer]
        for call, response in zip(model_calls, ("Hello", "Hello world"), strict=True):
            _, inputs, labels, grad_enabled, training = call
            ids = document_tokens(tokenizer, "Question", response)
            assert inputs.tolist() == [ids[:-1]]
            assert labels.tolist() == [ids[1:]]
            assert labels[0, -1] == tokenizer.eos_token_id
            assert grad_enabled == (scorer is model)
            assert not training
    assert model.training and reference.training


def test_dpo_identical_initial_models_have_log_two_loss_but_nonzero_gradient(dpo_models, tokenizer):
    model, _ = dpo_models
    reference = deepcopy(model)
    first = compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "world")
    second = compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "world")
    torch.testing.assert_close(first, torch.tensor(math.log(2)))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.backward()
    assert any(parameter.grad.abs().sum() > 0 for parameter in model.parameters() if parameter.grad is not None)
    assert all(parameter.grad is None for parameter in reference.parameters())


@pytest.mark.parametrize("training, ref_training", [(True, True), (True, False), (False, True), (False, False)])
def test_dpo_restores_modes_after_success_and_failure(dpo_models, tokenizer, monkeypatch, training, ref_training):
    model, reference = dpo_models
    model.train(training)
    reference.train(ref_training)
    compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "world")
    assert (model.training, reference.training) == (training, ref_training)
    monkeypatch.setattr(dpo, "get_response_log_probs", Mock(side_effect=RuntimeError("scoring failed")))
    with pytest.raises(RuntimeError, match="scoring failed"):
        compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "world")
    assert (model.training, reference.training) == (training, ref_training)


def test_dpo_preserves_existing_reference_gradients(dpo_models, tokenizer):
    model, reference = dpo_models
    for parameter in reference.parameters():
        parameter.grad = torch.ones_like(parameter)
    compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "world").backward()
    for parameter in reference.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)


@pytest.mark.parametrize("chosen, rejected", [("", "world"), ("Hello", ""), ("", "")])
def test_dpo_accepts_empty_responses_and_still_scores_eos(dpo_models, tokenizer, chosen, rejected):
    model, reference = dpo_models
    loss = compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", chosen, rejected)
    assert torch.isfinite(loss)
    loss.backward()
    if chosen == rejected:
        torch.testing.assert_close(loss, torch.tensor(math.log(2)))


@pytest.mark.parametrize("beta", [-0.1, float("nan"), float("inf")])
def test_dpo_rejects_invalid_beta_before_scoring(dpo_models, tokenizer, beta, monkeypatch):
    score = Mock()
    monkeypatch.setattr(dpo, "get_response_log_probs", score)
    with pytest.raises(ValueError, match="beta"):
        compute_per_instance_dpo_loss(*dpo_models, tokenizer, beta, "Question", "Hello", "world")
    score.assert_not_called()


def test_dpo_rejects_missing_eos_and_nonstring_responses(dpo_models, tokenizer):
    with pytest.raises(TypeError, match="strings"):
        compute_per_instance_dpo_loss(*dpo_models, tokenizer, 0.1, "Question", None, "world")
    tokenizer.eos_token = None
    with pytest.raises(ValueError, match="eos_token_id"):
        compute_per_instance_dpo_loss(*dpo_models, tokenizer, 0.1, "Question", "Hello", "world")


@pytest.mark.parametrize("chosen_score", [-10000.0, 10000.0])
def test_dpo_logsigmoid_is_stable_for_extreme_margins(dpo_models, tokenizer, monkeypatch, chosen_score):
    model, reference = dpo_models

    def score(scorer, ids):
        if scorer is reference:
            return torch.tensor(0.0)
        value = chosen_score if ids[-2] == tokenizer.convert_tokens_to_ids("Hello") else -chosen_score
        return next(model.parameters()).reshape(-1)[0] * 0 + value

    monkeypatch.setattr(dpo, "_sequence_log_prob", score)
    loss = compute_per_instance_dpo_loss(model, reference, tokenizer, 1.0, "Question", "Hello", "world")
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(20000.0 if chosen_score < 0 else 0.0)
    loss.backward()


def test_dpo_zero_beta_has_constant_loss_and_zero_policy_gradients(dpo_models, tokenizer):
    model, reference = dpo_models
    loss = compute_per_instance_dpo_loss(model, reference, tokenizer, 0.0, "Question", "Hello", "world")
    torch.testing.assert_close(loss, torch.tensor(math.log(2)))
    loss.backward()
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_dpo_supports_policy_and_reference_on_different_devices(dpo_models, tokenizer):
    model, reference = dpo_models
    model.to("cuda:0")
    reference.to("cuda:1")
    loss = compute_per_instance_dpo_loss(model, reference, tokenizer, 0.1, "Question", "Hello", "world")
    assert loss.device == torch.device("cuda:0")
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())
