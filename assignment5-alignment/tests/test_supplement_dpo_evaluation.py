from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from cs336_alignment.supplement import dpo_training
from cs336_alignment.supplement.dpo import compute_per_instance_dpo_loss
from cs336_alignment.supplement.dpo_training import dpo_train_step, evaluate_dpo
from cs336_alignment.supplement.hh import HHPreferenceExample


@pytest.fixture
def models(tokenizer):
    torch.manual_seed(14)
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
    return [HHPreferenceExample("Question", chosen, rejected, "test") for chosen, rejected in [
        ("Hello", "Hello world"), ("Hello world", "Hello"), ("", ""),
    ]]


def raw_margin(model, tokenizer, example):
    scores = []
    for response in (example.chosen, example.rejected):
        document = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example.instruction}\n\n### Response:\n{response}"
        )
        ids = torch.tensor(tokenizer.encode(document) + [tokenizer.eos_token_id])
        logits = model(input_ids=ids[:-1].unsqueeze(0)).logits.squeeze(0)
        scores.append(-F.cross_entropy(logits.float(), ids[1:], reduction="sum"))
    return scores[0] - scores[1]


@pytest.mark.parametrize("training, ref_training", [(True, True), (True, False), (False, True), (False, False)])
def test_dpo_evaluation_matches_scores_and_preserves_model_state(models, tokenizer, examples, training, ref_training):
    policy, reference = models
    expected_policy, expected_ref = deepcopy(policy).eval(), deepcopy(reference).eval()
    with torch.no_grad():
        expected_loss = torch.stack([
            compute_per_instance_dpo_loss(expected_policy, expected_ref, tokenizer, 0.1,
                                          example.instruction, example.chosen, example.rejected)
            for example in examples
        ]).mean().item()
        expected_accuracy = sum(raw_margin(expected_policy, tokenizer, example).item() > 0 for example in examples) / len(examples)
    policy.train(training)
    reference.train(ref_training)
    before = [[parameter.detach().clone() for parameter in model.parameters()] for model in models]
    for model in models:
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
    calls = []

    def inspect(model, args):
        calls.append((model.training, torch.is_grad_enabled()))

    handles = [model.register_forward_pre_hook(inspect) for model in models]
    try:
        actual = evaluate_dpo(policy, reference, tokenizer, iter(examples))
    finally:
        for handle in handles:
            handle.remove()
    assert actual["val_loss"] == pytest.approx(expected_loss)
    assert actual["val_accuracy"] == expected_accuracy == 1 / 3
    assert actual["num_examples"] == 3
    assert len(calls) == 4 * len(examples)
    assert all(call == (False, False) for call in calls)
    assert (policy.training, reference.training) == (training, ref_training)
    assert torch.is_grad_enabled()
    for model, originals in zip(models, before, strict=True):
        for parameter, original in zip(model.parameters(), originals, strict=True):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)
            assert parameter.requires_grad


def test_dpo_accuracy_is_raw_policy_preference_not_reference_adjusted(models, tokenizer, examples, monkeypatch):
    scores = iter([(1.0, 1.0, 2.0), (3.0, -1.0, -2.0), (5.0, 0.0, -1.0)])
    monkeypatch.setattr(dpo_training, "compute_dpo_loss_and_margins", lambda *args: tuple(torch.tensor(value) for value in next(scores)))
    actual = evaluate_dpo(*models, tokenizer, examples)
    assert actual == {"val_loss": 3.0, "val_accuracy": 1 / 3, "num_examples": 3}


def test_empty_dpo_validation_raises_instead_of_returning_zero(models, tokenizer):
    with pytest.raises(ValueError, match="at least one preference example"):
        evaluate_dpo(*models, tokenizer, [])


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("failure", ["source", "bad_example"])
def test_dpo_validation_failure_preserves_modes(models, tokenizer, examples, training, failure):
    policy, reference = models
    policy.train(training)
    reference.train(not training)

    def source():
        yield examples[0]
        if failure == "source":
            raise RuntimeError("read failed")
        yield HHPreferenceExample("Question", None, "Hello", "test")

    with pytest.raises(RuntimeError if failure == "source" else TypeError):
        evaluate_dpo(policy, reference, tokenizer, source())
    assert (policy.training, reference.training) == (training, not training)
    assert all(parameter.grad is None for model in models for parameter in model.parameters())
    assert torch.is_grad_enabled()


def test_dpo_training_continues_after_validation(models, tokenizer, examples):
    policy, reference = models
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-4)
    dpo_train_step(policy, reference, tokenizer, optimizer, examples)
    metrics = evaluate_dpo(policy, reference, tokenizer, examples)
    assert policy.training and not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    dpo_train_step(policy, reference, tokenizer, optimizer, examples)
    assert any(not torch.equal(parameter, old) for parameter, old in zip(policy.parameters(), before, strict=True))
    assert metrics["num_examples"] == 3
