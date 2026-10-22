from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cs336_alignment.supplement import dpo_training
from cs336_alignment.supplement.dpo import compute_per_instance_dpo_loss
from cs336_alignment.supplement.dpo_training import run_dpo_epoch
from cs336_alignment.supplement.hh import HHPreferenceExample


@pytest.fixture
def models(tokenizer):
    torch.manual_seed(13)
    policy = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_ctx=128,
        n_embd=8, n_layer=1, n_head=2, n_inner=16,
        resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2, use_cache=False,
    ))
    return policy, deepcopy(policy)


@pytest.fixture
def examples():
    return [
        HHPreferenceExample("Question", chosen, rejected, f"source_{index}")
        for index, (chosen, rejected) in enumerate([
            ("Hello", "world"), ("This is a test", "Hello"), ("Answer", ""),
            ("Hello world", "another test"), ("Response", "This is another test"),
        ])
    ]


@pytest.mark.parametrize("accumulation", [1, 2, 5, 8])
def test_dpo_epoch_matches_sequential_mean_loss_updates(models, tokenizer, examples, accumulation):
    policy, reference = models
    expected_policy = deepcopy(policy)
    expected_reference = deepcopy(reference).requires_grad_(False).eval()
    expected_optimizer = torch.optim.RMSprop(expected_policy.parameters(), lr=1e-4)
    expected_metrics = []
    for start in range(0, len(examples), accumulation):
        group = examples[start:start + accumulation]
        expected_optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([
            compute_per_instance_dpo_loss(
                expected_policy, expected_reference, tokenizer, 0.1,
                example.instruction, example.chosen, example.rejected,
            ) for example in group
        ]).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(expected_policy.parameters(), 0.05)
        expected_optimizer.step()
        expected_metrics.append((loss.item(), norm.item(), len(group)))

    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-4)
    optimizer.step = Mock(wraps=optimizer.step)
    before_reference = [parameter.detach().clone() for parameter in reference.parameters()]
    results = list(run_dpo_epoch(policy, reference, tokenizer, optimizer, iter(examples), accumulation, max_grad_norm=0.05))
    assert optimizer.step.call_count == len(expected_metrics)
    assert [update for update, _ in results] == list(range(1, len(expected_metrics) + 1))
    assert sum(metrics["num_examples"] for _, metrics in results) == len(examples)
    for (_, metrics), (loss, norm, count) in zip(results, expected_metrics, strict=True):
        assert metrics["loss"] == pytest.approx(loss, rel=1e-4, abs=1e-6)
        assert metrics["grad_norm"] == pytest.approx(norm, rel=1e-4, abs=1e-6)
        assert metrics["num_examples"] == count
        assert metrics["learning_rate"] == 1e-4
        assert all(isinstance(value, (int, float)) for value in metrics.values())
    for actual, expected in zip(policy.parameters(), expected_policy.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        assert actual.grad is None
    for actual, before in zip(reference.parameters(), before_reference, strict=True):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
        assert not actual.requires_grad and actual.grad is None


def test_dpo_epoch_reads_only_one_group_per_request(models, tokenizer, examples, monkeypatch):
    policy, reference = models
    consumed = []

    def source():
        for index, example in enumerate(examples):
            consumed.append(index)
            yield example

    step = Mock(wraps=dpo_training.dpo_train_step)
    monkeypatch.setattr(dpo_training, "dpo_train_step", step)
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    updates = run_dpo_epoch(policy, reference, tokenizer, optimizer, source(), 2)
    assert consumed == []
    step.assert_not_called()
    for update, expected_indices in enumerate(([0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]), start=1):
        number, metrics = next(updates)
        assert number == update
        assert consumed == expected_indices
        assert optimizer.step.call_count == update
        assert metrics["num_examples"] == (1 if update == 3 else 2)
    with pytest.raises(StopIteration):
        next(updates)
    groups = [call.args[4] for call in step.call_args_list]
    assert [example for group in groups for example in group] == examples


@pytest.mark.parametrize("accumulation", [0, -1, 1.5])
def test_dpo_epoch_rejects_invalid_accumulation_before_reading(models, tokenizer, examples, accumulation):
    policy, reference = models
    consumed = []

    def source():
        consumed.append(True)
        yield from examples

    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    with pytest.raises(ValueError, match="positive integer"):
        list(run_dpo_epoch(policy, reference, tokenizer, optimizer, source(), accumulation))
    assert consumed == []
    optimizer.step.assert_not_called()


def test_empty_dpo_epoch_has_no_side_effects(models, tokenizer):
    policy, reference = models
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    optimizer.step = Mock(wraps=optimizer.step)
    assert list(run_dpo_epoch(policy, reference, tokenizer, optimizer, [], 2)) == []
    optimizer.step.assert_not_called()
    assert all(parameter.requires_grad for parameter in reference.parameters())


def test_dpo_epoch_does_not_yield_or_read_ahead_after_failed_group(models, tokenizer, examples, monkeypatch):
    policy, reference = models
    consumed = []

    def source():
        for index, example in enumerate(examples):
            consumed.append(index)
            yield example

    optimizer = torch.optim.RMSprop(policy.parameters(), lr=1e-6)
    monkeypatch.setattr(dpo_training, "dpo_train_step", Mock(side_effect=RuntimeError("group failed")))
    updates = run_dpo_epoch(policy, reference, tokenizer, optimizer, source(), 2)
    with pytest.raises(RuntimeError, match="group failed"):
        next(updates)
    assert consumed == [0, 1]
