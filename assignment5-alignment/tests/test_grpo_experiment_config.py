from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cs336_alignment.grpo_experiment import DATA_DIR, GRPOExperimentConfig


def test_defaults_match_the_standard_experiment():
    config = GRPOExperimentConfig()
    assert config.model_id == "allenai/OLMo-2-0425-1B"
    assert config.prompt_name == "r1_zero"
    assert config.train_path == DATA_DIR / "train.jsonl"
    assert config.val_path == DATA_DIR / "test.jsonl"
    assert config.n_train_examples == 6400
    assert config.n_val_examples == 1024
    assert config.num_rollout_steps == 200
    assert config.rollout_batch_size == config.train_batch_size == 256
    assert config.prompts_per_rollout_batch == 32
    assert config.gradient_accumulation_steps == 32
    assert config.learning_rate == 1e-5
    assert config.sampling_temperature == 1.0
    assert config.sampling_max_tokens == 512
    assert config.max_grad_norm == 1.0
    assert config.eval_interval == 10
    assert config.rollout_log_interval == 40
    assert config.importance_reweighting_method == "none"
    assert config.cliprange is None


def test_small_configuration_does_not_load_files_or_a_model(tmp_path):
    config = GRPOExperimentConfig(
        model_id="local/model", train_path=tmp_path / "train.jsonl", val_path=tmp_path / "val.jsonl",
        n_train_examples=4, n_val_examples=2, num_rollout_steps=1,
        rollout_batch_size=6, group_size=2, gradient_accumulation_steps=4,
        max_grad_norm=None,
    )
    assert isinstance(config.train_path, Path)
    assert not config.train_path.exists()
    assert config.prompts_per_rollout_batch == 3
    assert config.train_batch_size == 6
    # Unequal microbatch sizes are supported by the existing accumulation helper.
    assert config.gradient_accumulation_steps == 4


def test_configuration_cannot_be_mutated_after_creation():
    config = GRPOExperimentConfig()
    with pytest.raises(FrozenInstanceError):
        config.group_size = 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_train_examples": 0}, {"n_val_examples": -1}, {"num_rollout_steps": 0},
        {"rollout_batch_size": 0}, {"group_size": 0}, {"rollout_batch_size": 255},
        {"gradient_accumulation_steps": 0}, {"gradient_accumulation_steps": 257},
        {"sampling_max_tokens": 0}, {"eval_interval": 0}, {"rollout_log_interval": 0},
        {"learning_rate": 0.0}, {"learning_rate": float("nan")},
        {"sampling_temperature": -1.0}, {"sampling_temperature": float("inf")},
        {"max_grad_norm": -1.0}, {"max_grad_norm": float("nan")},
        {"model_id": " "}, {"prompt_name": "r1_zero_three_shot"},
        {"importance_reweighting_method": "gspo"},
        {"importance_reweighting_method": "grpo"},
        {"importance_reweighting_method": "grpo", "cliprange": -0.1},
        {"importance_reweighting_method": "grpo", "cliprange": float("nan")},
        {"importance_reweighting_method": "grpo", "cliprange": float("inf")},
    ],
)
def test_rejects_invalid_experiment_settings(kwargs):
    with pytest.raises(ValueError):
        GRPOExperimentConfig(**kwargs)


@pytest.mark.parametrize("method, cliprange", [("noclip", None), ("grpo", 0.2), ("grpo", 0.0)])
def test_configuration_accepts_token_reweighting(method, cliprange):
    config = GRPOExperimentConfig(importance_reweighting_method=method, cliprange=cliprange)
    assert config.importance_reweighting_method == method
    assert config.cliprange == cliprange
