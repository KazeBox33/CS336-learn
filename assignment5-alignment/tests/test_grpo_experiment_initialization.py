import random

import numpy as np
import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    initialize_training_components,
)
from cs336_alignment.policy import get_response_log_probs


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_initialization_uses_config_and_prepares_training(
    monkeypatch, tiny_train_model, tokenizer, device
):
    calls = []

    def load_model(model_id, requested_device):
        calls.append((model_id, requested_device))
        return tiny_train_model, tokenizer

    monkeypatch.setattr(grpo_experiment, "get_model_and_tokenizer", load_model)
    tiny_train_model.eval()
    tiny_train_model.config.use_cache = True
    config = GRPOExperimentConfig(model_id="local/model", learning_rate=2e-5)

    model, returned_tokenizer, optimizer = initialize_training_components(config, device)

    assert calls == [("local/model", device)]
    assert model is tiny_train_model
    assert returned_tokenizer is tokenizer
    assert model.training
    assert model.config.use_cache is False
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 2e-5
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["weight_decay"] == 0.0
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }


def test_seed_is_applied_before_model_loading(monkeypatch, tiny_train_model, tokenizer):
    draws = []

    def load_model(_model_id, _device):
        draws.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
        return tiny_train_model, tokenizer

    monkeypatch.setattr(grpo_experiment, "get_model_and_tokenizer", load_model)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        for seed in (7, 7, 8):
            initialize_training_components(GRPOExperimentConfig(seed=seed), "cpu")
        assert draws[0] == draws[1]
        assert draws[0] != draws[2]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def test_initialized_optimizer_updates_the_loaded_model(monkeypatch, tiny_train_model, tokenizer):
    monkeypatch.setattr(
        grpo_experiment, "get_model_and_tokenizer", lambda *_: (tiny_train_model, tokenizer)
    )
    model, _, optimizer = initialize_training_components(GRPOExperimentConfig(), "cpu")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    scores = get_response_log_probs(
        model, torch.tensor([[3, 4]]), torch.tensor([[4, 5]])
    )
    (-scores["log_probs"].mean()).backward()
    optimizer.step()

    assert any(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, model.parameters(), strict=True)
    )
