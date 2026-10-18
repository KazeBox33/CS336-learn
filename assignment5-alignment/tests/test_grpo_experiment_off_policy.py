import json
import random
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import GRPOExperimentConfig, GRPOExperimentData
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.rewards import compute_group_normalized_rewards
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


@pytest.fixture
def data():
    return GRPOExperimentData(
        [GSM8KExample("Q0", "0"), GSM8KExample("Q1", "1")],
        [GSM8KExample("V0", "0")], ["Hello", "This is"], ["Hello"],
    )


@pytest.mark.parametrize("method", ["none", "noclip", "grpo", "gspo"])
def test_iterations_prepare_and_cache_once_then_update_each_minibatch(
    monkeypatch, tiny_train_model, tokenizer, data, method
):
    config = GRPOExperimentConfig(
        rollout_batch_size=6, train_batch_size=2, group_size=3, gradient_accumulation_steps=1,
        importance_reweighting_method=method, cliprange=3e-4 if method == "gspo" else 0.2,
    )
    policy = tiny_train_model
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = [
        VLLMCompletion(text, [], "stop") for text in ["world", "test", "a test"] * 2
    ]
    snapshots = []
    server.sync_policy_weights.side_effect = lambda model: snapshots.append(
        [parameter.detach().clone() for parameter in model.parameters()]
    )

    def grade(response, truth):
        reward = {"world": 1.0, "test": 0.0, "a test": 0.5}[response]
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    reward_fn = Mock(side_effect=grade)
    prepare = Mock(wraps=grpo_experiment._prepare_rollout_batch)
    train = Mock(wraps=grpo_experiment.train_rollout_batch)
    original_cache = grpo_experiment.get_old_log_probs
    caches = []

    def cache(*args, **kwargs):
        assert optimizer.step.call_count == len(caches) * 3
        assert kwargs["microbatch_size"] == 2
        scores = original_cache(*args, **kwargs)
        caches.append((scores, scores.clone()))
        return scores

    cache_mock = Mock(side_effect=cache)
    monkeypatch.setattr(grpo_experiment, "_prepare_rollout_batch", prepare)
    monkeypatch.setattr(grpo_experiment, "train_rollout_batch", train)
    monkeypatch.setattr(grpo_experiment, "get_old_log_probs", cache_mock)
    single_step = Mock(side_effect=AssertionError("Do not regroup string responses per minibatch."))
    monkeypatch.setattr(grpo_experiment, "grpo_train_step", single_step)
    expected_advantages, _ = compute_group_normalized_rewards(torch.tensor([1.0, 0.0, 0.5] * 2), 3)

    rng = random.Random(0)
    for iteration in range(2):
        metrics, prompts, completions = grpo_experiment.run_grpo_iteration(
            config, data, server, policy, tokenizer, optimizer, reward_fn, rng=rng
        )
        assert len(prompts) == len(completions) == 6
        assert metrics["optimizer_steps"] == 3
        assert metrics["mean_reward"] == 0.5
        assert optimizer.step.call_count == (iteration + 1) * 3
        kwargs = train.call_args.kwargs
        torch.testing.assert_close(kwargs["advantages"], expected_advantages)
        assert kwargs["token_batch"]["input_ids"].shape[0] == 6
        assert kwargs["old_log_probs"] is (None if method == "none" else caches[-1][0])

    assert prepare.call_count == train.call_count == server.sync_policy_weights.call_count == 2
    assert server.generate_completions.call_count == 2
    assert reward_fn.call_count == 12
    assert cache_mock.call_count == (0 if method == "none" else 2)
    single_step.assert_not_called()
    assert any(not torch.equal(before, after) for before, after in zip(*snapshots, strict=True))
    for scores, saved in caches:
        torch.testing.assert_close(scores, saved)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_handout_iteration_generates_once_and_takes_32_steps(tiny_train_model, tokenizer, data):
    config = GRPOExperimentConfig(
        train_batch_size=8, gradient_accumulation_steps=1,
        importance_reweighting_method="gspo", cliprange=3e-4,
    )
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = [
        VLLMCompletion(text, [], "stop") for text in ["world", "test"] * 128
    ]

    def grade(response, truth):
        reward = float(response == "world")
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    reward_fn = Mock(side_effect=grade)
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    optimizer.step = Mock(wraps=optimizer.step)
    metrics, _, _ = grpo_experiment.run_grpo_iteration(
        config, data, server, tiny_train_model, tokenizer, optimizer, reward_fn, rng=random.Random(0)
    )
    server.sync_policy_weights.assert_called_once()
    server.generate_completions.assert_called_once()
    assert optimizer.step.call_count == metrics["optimizer_steps"] == 32
    assert reward_fn.call_count == 256
    assert metrics["mean_reward"] == 0.5


def test_rollout_logs_count_optimizer_updates_not_just_generation_batches(monkeypatch, tmp_path, data):
    config = GRPOExperimentConfig(
        num_rollout_steps=2, train_batch_size=8, gradient_accumulation_steps=1,
        eval_interval=1, rollout_log_interval=1,
    )
    completions = [VLLMCompletion("answer", [], "stop")]
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", Mock(return_value=(
        {"loss": 0.5, "optimizer_steps": 32}, ["Hello"], completions
    )))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", Mock(return_value=(
        {"val_accuracy": 0.0}, completions
    )))
    callback = Mock()
    results = list(grpo_experiment.run_grpo_training_loop(
        config, data, *[Mock() for _ in range(5)], rollout_output_dir=tmp_path, on_step=callback
    ))
    assert [step for step, _ in results] == [0, 1, 2]
    assert [call.args[0] for call in callback.call_args_list] == [1, 2]
    for step in (1, 2):
        assert results[step][1]["train_optimizer_steps"] == 32
        for split in ("train", "val"):
            path = tmp_path / f"{split}_step_{step:04d}.jsonl"
            record = json.loads(path.read_text().splitlines()[0])
            assert record["step"] == step
            assert record["policy_step"] == (step - (split == "train")) * 32
