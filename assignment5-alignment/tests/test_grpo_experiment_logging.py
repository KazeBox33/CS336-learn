import json
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    run_grpo_training_loop,
    write_grpo_metrics,
)


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_record_per_step_and_creates_parent_directory(tmp_path):
    path = tmp_path / "seed_0" / "metrics.jsonl"
    steps = [
        (0, {"val_accuracy": 0.125}),
        (1, {"train_loss": -0.1, "train_mean_reward": 0.25}),
        (2, {"train_loss": -0.2, "val_accuracy": 0.25}),
    ]
    assert write_grpo_metrics(iter(steps), str(path)) == 3
    assert read_records(path) == [{"step": step, **metrics} for step, metrics in steps]
    assert all("step" not in metrics for _, metrics in steps)
    assert path.read_bytes().endswith(b"\n")


def test_existing_file_is_not_overwritten_and_training_does_not_start(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("previous experiment\n", encoding="utf-8")
    started = Mock()

    def steps():
        started()
        yield 0, {"val_accuracy": 0.0}

    with pytest.raises(FileExistsError):
        write_grpo_metrics(steps(), path)
    assert path.read_text(encoding="utf-8") == "previous experiment\n"
    started.assert_not_called()


def test_each_record_is_visible_before_the_next_step_starts(tmp_path):
    path = tmp_path / "metrics.jsonl"

    def steps():
        yield 0, {"val_accuracy": 0.125}
        assert read_records(path) == [{"step": 0, "val_accuracy": 0.125}]
        yield 1, {"train_loss": 0.5}
        assert len(read_records(path)) == 2

    assert write_grpo_metrics(steps(), path) == 2


def test_generator_failure_preserves_completed_records_and_propagates(tmp_path):
    path = tmp_path / "metrics.jsonl"

    def steps():
        yield 0, {"val_accuracy": 0.125}
        yield 1, {"train_loss": 0.5}
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        write_grpo_metrics(steps(), path)
    assert read_records(path) == [
        {"step": 0, "val_accuracy": 0.125},
        {"step": 1, "train_loss": 0.5},
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_are_rejected_before_writing_the_record(tmp_path, value):
    path = tmp_path / "metrics.jsonl"
    steps = [(0, {"val_accuracy": 0.125}), (1, {"train_loss": value})]
    with pytest.raises(ValueError, match="JSON compliant"):
        write_grpo_metrics(steps, path)
    assert read_records(path) == [{"step": 0, "val_accuracy": 0.125}]


def test_metrics_cannot_replace_the_step_number(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(ValueError, match="reserved key"):
        write_grpo_metrics([(1, {"step": 99, "train_loss": 0.5})], path)
    assert path.read_text(encoding="utf-8") == ""


def test_writer_consumes_training_loop_and_preserves_sparse_validation_metrics(monkeypatch, tmp_path):
    config = GRPOExperimentConfig(num_rollout_steps=3, eval_interval=2)
    train = Mock(return_value=({"loss": torch.tensor(0.5), "mean_reward": 0.25}, [], []))
    validate = Mock(side_effect=[
        ({"val_accuracy": 0.125}, []),
        ({"val_accuracy": 0.25}, []),
        ({"val_accuracy": 0.375}, []),
    ])
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train)
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate)
    path = tmp_path / "metrics.jsonl"
    loop = run_grpo_training_loop(config, *[Mock() for _ in range(6)])
    assert write_grpo_metrics(loop, path) == 4
    records = read_records(path)
    assert [record["optimizer_updates"] for record in records] == [0, 1, 2, 3]
    assert all(record["elapsed_seconds"] >= 0 for record in records)
    assert [record["step"] for record in records if "val_seconds" in record] == [0, 2, 3]
    stripped = [{key: value for key, value in record.items() if key not in {
        "elapsed_seconds", "optimizer_updates", "train_iteration_seconds", "val_seconds",
    }} for record in records]
    assert stripped == [
        {"step": 0, "val_accuracy": 0.125},
        {"step": 1, "train_loss": 0.5, "train_mean_reward": 0.25},
        {"step": 2, "train_loss": 0.5, "train_mean_reward": 0.25, "val_accuracy": 0.25},
        {"step": 3, "train_loss": 0.5, "train_mean_reward": 0.25, "val_accuracy": 0.375},
    ]
    assert train.call_count == validate.call_count == 3
