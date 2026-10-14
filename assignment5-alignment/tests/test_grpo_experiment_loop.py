import random
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    GRPOExperimentData,
    run_grpo_training_loop,
)
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


@pytest.mark.parametrize(
    "num_steps, interval, validation_steps",
    [(5, 2, [0, 2, 4, 5]), (4, 2, [0, 2, 4]), (1, 10, [0, 1])],
)
def test_loop_schedules_validation_and_yields_detached_scalars(
    monkeypatch, num_steps, interval, validation_steps
):
    config = GRPOExperimentConfig(num_rollout_steps=num_steps, eval_interval=interval, seed=7)
    data, server, policy, tokenizer, optimizer, reward_fn = [Mock() for _ in range(6)]
    events, rngs, random_values = [], [], []

    def train(*args, rng):
        assert args == (config, data, server, policy, tokenizer, optimizer, reward_fn)
        rngs.append(rng)
        random_values.append(rng.random())
        step = len(rngs)
        events.append(("train", step))
        return {"loss": torch.tensor(float(step), requires_grad=True), "mean_reward": 0.5}, [], []

    def validate(*args, batch_size):
        assert args == (config, data, server, policy, reward_fn)
        assert batch_size == 3
        events.append(("val", len(rngs)))
        return {"val_accuracy": 0.25}, []

    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate)
    results = list(run_grpo_training_loop(
        config, data, server, policy, tokenizer, optimizer, reward_fn, validation_batch_size=3
    ))

    assert [step for step, _ in results] == list(range(num_steps + 1))
    assert results[0] == (0, {"val_accuracy": 0.25})
    assert [step for event, step in events if event == "val"] == validation_steps
    expected_events = [("val", 0)]
    for step in range(1, num_steps + 1):
        expected_events.append(("train", step))
        if step in validation_steps:
            expected_events.append(("val", step))
        expected = {"train_loss": float(step), "train_mean_reward": 0.5}
        if step in validation_steps:
            expected["val_accuracy"] = 0.25
        assert results[step] == (step, expected)
        assert all(isinstance(value, (int, float)) for value in results[step][1].values())
    assert events == expected_events
    assert all(rng is rngs[0] for rng in rngs)
    expected_rng = random.Random(config.seed)
    assert random_values == [expected_rng.random() for _ in range(num_steps)]


def test_loop_waits_for_the_caller_before_starting_the_next_update(monkeypatch):
    config = GRPOExperimentConfig(num_rollout_steps=2, eval_interval=10)
    validate = Mock(return_value=({"val_accuracy": 0.0}, []))
    train = Mock(return_value=({"loss": torch.tensor(0.1)}, [], []))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate)
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    loop = run_grpo_training_loop(config, *[Mock() for _ in range(6)])
    validate.assert_not_called()
    train.assert_not_called()
    assert next(loop) == (0, {"val_accuracy": 0.0})
    train.assert_not_called()
    assert next(loop)[0] == 1
    assert train.call_count == 1
    assert next(loop)[0] == 2
    assert train.call_count == 2
    with pytest.raises(StopIteration):
        next(loop)


def test_failed_initial_validation_prevents_training(monkeypatch):
    validate = Mock(side_effect=RuntimeError("validation failed"))
    train = Mock()
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate)
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    with pytest.raises(RuntimeError, match="validation failed"):
        list(run_grpo_training_loop(GRPOExperimentConfig(), *[Mock() for _ in range(6)]))
    train.assert_not_called()


def test_failed_training_stops_the_loop_without_extra_validation(monkeypatch):
    validate = Mock(return_value=({"val_accuracy": 0.0}, []))
    train = Mock(side_effect=RuntimeError("training failed"))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate)
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    loop = run_grpo_training_loop(GRPOExperimentConfig(), *[Mock() for _ in range(6)])
    next(loop)
    with pytest.raises(RuntimeError, match="training failed"):
        next(loop)
    assert train.call_count == validate.call_count == 1


def test_failed_periodic_validation_is_not_reported_as_a_completed_step(monkeypatch):
    validate = Mock(side_effect=[({"val_accuracy": 0.0}, []), RuntimeError("validation failed")])
    train = Mock(return_value=({"loss": torch.tensor(0.1)}, [], []))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate)
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    config = GRPOExperimentConfig(num_rollout_steps=2, eval_interval=1)
    loop = run_grpo_training_loop(config, *[Mock() for _ in range(6)])
    next(loop)
    with pytest.raises(RuntimeError, match="validation failed"):
        next(loop)
    assert train.call_count == 1


def test_loop_connects_real_cpu_updates_to_validation(tiny_train_model, tokenizer):
    config = GRPOExperimentConfig(
        num_rollout_steps=2, rollout_batch_size=4, group_size=2,
        gradient_accumulation_steps=2, eval_interval=1,
    )
    data = GRPOExperimentData(
        train_examples=[GSM8KExample("Q0", "0"), GSM8KExample("Q1", "1")],
        train_prompts=["Hello", "This is"],
        val_examples=[GSM8KExample("V0", "0")],
        val_prompts=["Hello"],
    )
    policy = tiny_train_model
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.0)
    optimizer.step = Mock(wraps=optimizer.step)
    server = Mock(spec=VLLMServer)
    snapshots, events = [], []

    def sync(model):
        snapshots.append([p.detach().clone() for p in model.parameters()])
        events.append("sync")

    def generate(*, prompts, sampling_params, batch_size=None):
        assert events[-1] == "sync"
        n = sampling_params["n"]
        events.append("train" if n == 2 else "val")
        return [
            VLLMCompletion(text, [index], "stop")
            for _ in prompts
            for index, text in enumerate(["world", "test"][:n])
        ]

    def reward(response, truth):
        value = float(response == "world")
        return {"reward": value, "format_reward": 1.0, "answer_reward": value}

    server.sync_policy_weights.side_effect = sync
    server.generate_completions.side_effect = generate
    results = list(run_grpo_training_loop(config, data, server, policy, tokenizer, optimizer, reward))
    assert [step for step, _ in results] == [0, 1, 2]
    assert optimizer.step.call_count == 2
    assert events == ["sync", "val", "sync", "train", "sync", "val", "sync", "train", "sync", "val"]
    for _, metrics in results:
        assert metrics["val_accuracy"] == 1.0
        assert all(isinstance(value, (int, float)) for value in metrics.values())
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(snapshots[1], snapshots[2], strict=True))
    assert all(torch.equal(a, b) for a, b in zip(snapshots[2], snapshots[3], strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(snapshots[3], snapshots[4], strict=True))
    assert all(p.grad is None for p in policy.parameters())
