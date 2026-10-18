import random
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    GRPOExperimentData,
    run_grpo_iteration,
)
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.tokenization import tokenize_prompt_and_output
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


@pytest.fixture
def config():
    return GRPOExperimentConfig(
        rollout_batch_size=4, group_size=2, gradient_accumulation_steps=2,
        max_grad_norm=0.5,
    )


@pytest.fixture
def data():
    return GRPOExperimentData(
        train_examples=[GSM8KExample("Q0", "0"), GSM8KExample("Q1", "1")],
        val_examples=[],
        train_prompts=["Hello", "This is"],
        val_prompts=[],
    )


def test_iteration_connects_generation_to_exactly_one_training_step(monkeypatch, config, data):
    server = Mock(spec=VLLMServer)
    policy, tokenizer, optimizer, reward_fn = Mock(), Mock(), Mock(), Mock()
    rng = random.Random(0)
    prompts = ["Hello", "Hello", "This is", "This is"]
    truths = ["0", "0", "1", "1"]
    completions = [VLLMCompletion(text, [i], "stop") for i, text in enumerate(
        ["world", "test", "a test", "another test"]
    )]
    generate = Mock(return_value=(prompts, completions, truths))
    loss = torch.tensor(0.25)
    metadata = {"mean_reward": 0.5, "grad_norm": torch.tensor(0.1)}
    train = Mock(return_value=(loss, metadata))
    monkeypatch.setattr(grpo_experiment, "generate_training_rollouts", generate)
    monkeypatch.setattr(grpo_experiment, "grpo_train_step", train)
    cache = Mock(side_effect=AssertionError("Standard on-policy iterations must not cache old scores."))
    monkeypatch.setattr(grpo_experiment, "get_old_log_probs", cache)

    metrics, returned_prompts, returned_completions = run_grpo_iteration(
        config, data, server, policy, tokenizer, optimizer, reward_fn, rng=rng
    )

    generate.assert_called_once_with(config, data, server, policy, rng=rng)
    train.assert_called_once_with(
        model=policy, tokenizer=tokenizer, optimizer=optimizer,
        gradient_accumulation_steps=2, max_grad_norm=0.5, reward_fn=reward_fn,
        repeated_prompts=prompts,
        rollout_responses=[completion.text for completion in completions],
        repeated_ground_truths=truths, group_size=2,
        importance_reweighting_method="none", old_log_probs=None, cliprange=None,
    )
    assert metrics == {"loss": loss, **metadata}
    assert "loss" not in metadata
    assert returned_prompts is prompts
    assert returned_completions is completions
    cache.assert_not_called()


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
def test_iteration_caches_old_scores_after_generation_before_training(
    monkeypatch, config, data, tokenizer, method
):
    config = replace(config, importance_reweighting_method=method, cliprange=0.2)
    prompts = ["Hello", "Hello", "This is", "This is"]
    completions = [VLLMCompletion(text, [], "stop") for text in
                   ["world", "test", "a test", "another test"]]
    events = []
    policy, optimizer, reward_fn = Mock(), Mock(), Mock()
    expected_batch = tokenize_prompt_and_output(prompts, [item.text for item in completions], tokenizer)
    old_scores = torch.zeros_like(expected_batch["labels"], dtype=torch.float32)

    def generate(*args, **kwargs):
        events.append("generate")
        return prompts, completions, ["0", "0", "1", "1"]

    def cache(model, input_ids, labels, microbatch_size):
        assert events == ["generate"]
        assert model is policy
        assert microbatch_size == 2
        torch.testing.assert_close(input_ids, expected_batch["input_ids"])
        torch.testing.assert_close(labels, expected_batch["labels"])
        events.append("cache")
        return old_scores

    def train(**kwargs):
        assert events == ["generate", "cache"]
        assert kwargs["old_log_probs"] is old_scores
        assert kwargs["importance_reweighting_method"] == method
        assert kwargs["cliprange"] == 0.2
        events.append("train")
        return torch.tensor(0.0), {}

    monkeypatch.setattr(grpo_experiment, "generate_training_rollouts", generate)
    monkeypatch.setattr(grpo_experiment, "get_old_log_probs", cache)
    monkeypatch.setattr(grpo_experiment, "grpo_train_step", train)
    run_grpo_iteration(config, data, Mock(), policy, tokenizer, optimizer, reward_fn, rng=random.Random(0))
    assert events == ["generate", "cache", "train"]


def test_old_score_failure_prevents_training(monkeypatch, config, data, tokenizer):
    config = replace(config, importance_reweighting_method="noclip")
    generate = Mock(return_value=(["Hello"] * 4, [VLLMCompletion("world", [], "stop")] * 4, ["0"] * 4))
    cache = Mock(side_effect=RuntimeError("old scoring failed"))
    train = Mock()
    monkeypatch.setattr(grpo_experiment, "generate_training_rollouts", generate)
    monkeypatch.setattr(grpo_experiment, "get_old_log_probs", cache)
    monkeypatch.setattr(grpo_experiment, "grpo_train_step", train)
    with pytest.raises(RuntimeError, match="old scoring failed"):
        run_grpo_iteration(config, data, Mock(), Mock(), tokenizer, Mock(), Mock(), rng=random.Random(0))
    train.assert_not_called()


def test_generation_failure_prevents_training(monkeypatch, config, data):
    server = Mock(spec=VLLMServer)
    error = RuntimeError("generation failed")
    server.generate_completions.side_effect = error
    train = Mock()
    monkeypatch.setattr(grpo_experiment, "grpo_train_step", train)

    with pytest.raises(RuntimeError) as caught:
        run_grpo_iteration(config, data, server, Mock(), Mock(), Mock(), Mock(), rng=random.Random(0))

    assert caught.value is error
    train.assert_not_called()


def test_training_failure_is_not_swallowed(monkeypatch, config, data):
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = [VLLMCompletion("test", [], "stop")] * 4
    error = RuntimeError("training failed")
    train = Mock(side_effect=error)
    monkeypatch.setattr(grpo_experiment, "grpo_train_step", train)

    with pytest.raises(RuntimeError) as caught:
        run_grpo_iteration(config, data, server, Mock(), Mock(), Mock(), Mock(), rng=random.Random(0))

    assert caught.value is error
    train.assert_called_once()


@pytest.mark.parametrize("method", ["none", "noclip", "grpo", "gspo"])
def test_two_iterations_use_updated_weights_and_real_cpu_training(
    config, data, tiny_train_model, tokenizer, method
):
    config = replace(config, importance_reweighting_method=method, cliprange=0.2)
    policy = tiny_train_model
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.0)
    optimizer.step = Mock(wraps=optimizer.step)
    server = Mock(spec=VLLMServer)
    synced_parameters = []
    events = []

    def sync(model):
        assert model is policy
        synced_parameters.append([parameter.detach().clone() for parameter in model.parameters()])
        events.append("sync")

    def generate(*, prompts, sampling_params):
        assert events[-1] == "sync"
        assert sampling_params["n"] == 2
        events.append("generate")
        return [
            VLLMCompletion(text, [i], "stop")
            for _ in prompts
            for i, text in enumerate(["world", "test"])
        ]

    graded_pairs = []

    def reward(response, truth):
        graded_pairs.append((response, truth))
        value = float(response == "world")
        return {"reward": value, "format_reward": 1.0, "answer_reward": value}

    server.sync_policy_weights.side_effect = sync
    server.generate_completions.side_effect = generate
    rng = random.Random(0)
    for _ in range(2):
        metrics, prompts, completions = run_grpo_iteration(
            config, data, server, policy, tokenizer, optimizer, reward, rng=rng
        )
        assert len(prompts) == len(completions) == 4
        expected_truths = ["0" if prompt == "Hello" else "1" for prompt in prompts]
        assert graded_pairs[-4:] == list(zip([item.text for item in completions], expected_truths))
        assert metrics["mean_reward"] == 0.5
        assert metrics["mean_format_reward"] == 1.0
        for name in ("loss", "grad_norm", "mean_token_entropy"):
            assert torch.isfinite(metrics[name])
            assert not metrics[name].requires_grad
        assert all(parameter.grad is None for parameter in policy.parameters())

    assert optimizer.step.call_count == 2
    assert events == ["sync", "generate", "sync", "generate"]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(synced_parameters[0], synced_parameters[1], strict=True)
    )
    assert any(
        not torch.equal(before, parameter)
        for before, parameter in zip(synced_parameters[1], policy.parameters(), strict=True)
    )
