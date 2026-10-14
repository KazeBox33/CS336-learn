import json
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    GRPOExperimentData,
    run_grpo_training_loop,
    write_grpo_rollouts,
)
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.vllm_utils import VLLMCompletion


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_aligned_training_responses_and_records_pre_update_policy(tmp_path):
    root = tmp_path / "rollouts"
    completions = [VLLMCompletion("first\nanswer", [1, 2], "stop"), VLLMCompletion("second", [3], "length")]
    path = write_grpo_rollouts(40, "train", ["Q", "Q"], completions, str(root))
    assert path == root / "train_step_0040.jsonl"
    assert read_records(path) == [
        {
            "step": 40, "policy_step": 39, "split": "train", "response_index": index,
            "prompt": "Q", "response": completion.text, "token_ids": completion.token_ids,
            "finish_reason": completion.finish_reason,
        }
        for index, completion in enumerate(completions)
    ]
    assert path.read_bytes().endswith(b"\n")


def test_validation_records_use_current_policy_and_preserve_empty_responses(tmp_path):
    path = write_grpo_rollouts(0, "val", ["Q"], [VLLMCompletion("", [], None)], tmp_path)
    record = read_records(path)[0]
    assert record["policy_step"] == record["step"] == 0
    assert record["response"] == ""
    assert record["token_ids"] == []
    assert record["finish_reason"] is None


@pytest.mark.parametrize("prompts, completions", [([], []), (["Q"], [])])
def test_misaligned_or_empty_batches_do_not_create_files(tmp_path, prompts, completions):
    root = tmp_path / "rollouts"
    with pytest.raises(ValueError, match="non-empty and aligned"):
        write_grpo_rollouts(1, "train", prompts, completions, root)
    assert not root.exists()


@pytest.mark.parametrize("step, split", [(-1, "val"), (0, "train"), (1, "test")])
def test_invalid_step_or_split_is_rejected(tmp_path, step, split):
    root = tmp_path / "rollouts"
    with pytest.raises(ValueError):
        write_grpo_rollouts(step, split, ["Q"], [VLLMCompletion("A", [], "stop")], root)
    assert not root.exists()


def test_existing_snapshot_is_not_overwritten(tmp_path):
    original = VLLMCompletion("original", [1], "stop")
    path = write_grpo_rollouts(1, "train", ["Q"], [original], tmp_path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_grpo_rollouts(1, "train", ["Q"], [VLLMCompletion("replacement", [], "stop")], tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "num_steps, log_interval, train_steps, val_steps",
    [(5, 3, [1, 3, 5], [0, 2, 4, 5]), (4, 2, [1, 2, 4], [0, 2, 4]), (1, 40, [1], [0, 1])],
)
def test_loop_logs_existing_rollouts_without_extra_generation(
    monkeypatch, tmp_path, num_steps, log_interval, train_steps, val_steps
):
    config = GRPOExperimentConfig(
        num_rollout_steps=num_steps, eval_interval=2, rollout_log_interval=log_interval
    )
    data = GRPOExperimentData([], [GSM8KExample("V", "0")], [], ["VAL_PROMPT"])
    server, policy, tokenizer, optimizer, reward_fn = [Mock() for _ in range(5)]
    completed_steps = []

    def train(*args, rng):
        step = len(completed_steps) + 1
        completed_steps.append(step)
        completions = [VLLMCompletion(f"train {step}", [step], "stop") for _ in range(2)]
        return {"loss": torch.tensor(0.5)}, ["TRAIN_PROMPT"] * 2, completions

    def validate(*args, batch_size):
        step = len(completed_steps)
        return {"val_accuracy": 0.25}, [VLLMCompletion(f"val {step}", [step], "stop")]

    train_mock = Mock(side_effect=train)
    validate_mock = Mock(side_effect=validate)
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train_mock)
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate_mock)
    results = list(run_grpo_training_loop(
        config, data, server, policy, tokenizer, optimizer, reward_fn, rollout_output_dir=tmp_path
    ))
    assert [step for step, _ in results] == list(range(num_steps + 1))
    assert train_mock.call_count == num_steps
    assert validate_mock.call_count == len(val_steps)
    expected_files = {f"train_step_{step:04d}.jsonl" for step in train_steps}
    expected_files |= {f"val_step_{step:04d}.jsonl" for step in val_steps}
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    for split, steps in (("train", train_steps), ("val", val_steps)):
        for step in steps:
            records = read_records(tmp_path / f"{split}_step_{step:04d}.jsonl")
            assert len(records) == (2 if split == "train" else 1)
            assert all(record["response"] == f"{split} {step}" for record in records)
            assert all(record["policy_step"] == step - (split == "train") for record in records)
            assert all(record["prompt"] == ("TRAIN_PROMPT" if split == "train" else "VAL_PROMPT") for record in records)
    assert all(all(isinstance(value, (int, float)) for value in metrics.values()) for _, metrics in results)


def test_loop_does_not_write_rollouts_when_output_directory_is_not_set(monkeypatch):
    config = GRPOExperimentConfig(num_rollout_steps=1)
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", Mock(return_value=({"loss": 0.5}, [], [])))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", Mock(return_value=({"val_accuracy": 0.25}, [])))
    write = Mock()
    monkeypatch.setattr(grpo_experiment, "write_grpo_rollouts", write)
    results = list(run_grpo_training_loop(config, *[Mock() for _ in range(6)]))
    assert len(results) == 2
    write.assert_not_called()


def test_existing_initial_snapshot_stops_loop_without_training(monkeypatch, tmp_path):
    data = GRPOExperimentData([], [GSM8KExample("V", "0")], [], ["VAL_PROMPT"])
    completions = [VLLMCompletion("original", [], "stop")]
    path = write_grpo_rollouts(0, "val", data.val_prompts, completions, tmp_path)
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", Mock(return_value=({"val_accuracy": 0.0}, completions)))
    train = Mock()
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    with pytest.raises(FileExistsError):
        list(run_grpo_training_loop(
            GRPOExperimentConfig(), data, *[Mock() for _ in range(5)], rollout_output_dir=tmp_path
        ))
    train.assert_not_called()
    assert read_records(path)[0]["response"] == "original"
