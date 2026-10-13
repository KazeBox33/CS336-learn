from dataclasses import replace
import json

import pytest

from cs336_alignment.grpo_experiment import GRPOExperimentConfig, load_experiment_data
from cs336_alignment.gsm8k import GSM8KExample


def _write_dataset(path, questions):
    records = [
        {"question": question, "answer": f"Reasoning for example {i}.\n#### {i}"}
        for i, question in enumerate(questions)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


@pytest.fixture
def small_config(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_dataset(train_path, ["Train Q0?", "Train Q1?", "Train Q2?"])
    _write_dataset(val_path, ["Val Q0?", "Val Q1?", "Val Q2?"])
    return GRPOExperimentConfig(
        train_path=train_path, val_path=val_path, n_train_examples=2, n_val_examples=1
    )


def test_selects_requested_examples_and_keeps_prompts_aligned(small_config):
    data = load_experiment_data(small_config)
    assert data.train_examples == [GSM8KExample("Train Q0?", "0"), GSM8KExample("Train Q1?", "1")]
    assert data.val_examples == [GSM8KExample("Val Q0?", "0")]
    assert len(data.train_prompts) == 2
    assert len(data.val_prompts) == 1
    for example, prompt in zip(
        data.train_examples + data.val_examples, data.train_prompts + data.val_prompts, strict=True
    ):
        assert f"User: {example.question}\nAssistant: <think>" in prompt
        assert "{question}" not in prompt
        assert "Reasoning for example" not in prompt


def test_different_seeds_use_the_same_dataset_subsets(small_config):
    assert load_experiment_data(small_config) == load_experiment_data(replace(small_config, seed=123))


@pytest.mark.parametrize("field", ["n_train_examples", "n_val_examples"])
def test_rejects_requested_counts_larger_than_the_dataset(small_config, field):
    with pytest.raises(ValueError, match="Requested 4.*contains 3"):
        load_experiment_data(replace(small_config, **{field: 4}))


def test_rejects_identical_training_and_validation_files(small_config):
    with pytest.raises(ValueError, match="different dataset files"):
        load_experiment_data(replace(small_config, val_path=small_config.train_path))


def test_missing_dataset_file_is_reported(small_config, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_data(replace(small_config, train_path=tmp_path / "missing.jsonl"))


def test_loads_the_assignment_datasets_with_default_counts():
    data = load_experiment_data(GRPOExperimentConfig())
    assert len(data.train_examples) == len(data.train_prompts) == 6400
    assert len(data.val_examples) == len(data.val_prompts) == 1024
    assert data.train_examples[0].ground_truth == "72"
    assert data.val_examples[0].ground_truth == "18"
