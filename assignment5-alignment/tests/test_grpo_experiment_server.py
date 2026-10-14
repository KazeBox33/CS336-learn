from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import GRPOExperimentConfig, open_rollout_server
from cs336_alignment.vllm_utils import VLLMServer


@pytest.fixture
def policy():
    parameter = Mock(spec=torch.nn.Parameter)
    parameter.device = torch.device("cuda:0")
    model = Mock(spec=torch.nn.Module)
    model.parameters.side_effect = lambda: iter([parameter])
    return model


@pytest.fixture
def server_factory(monkeypatch):
    server = Mock(spec=VLLMServer)
    factory = Mock(return_value=server)
    monkeypatch.setattr(grpo_experiment, "VLLMServer", factory)
    return factory, server


def test_starts_syncs_and_stops_server_in_order(policy, server_factory):
    factory, server = server_factory
    events = []
    for name in ("start", "init_weight_sync", "sync_policy_weights", "stop"):
        getattr(server, name).side_effect = lambda *_, name=name: events.append(name)

    config = GRPOExperimentConfig(model_id="local/model", seed=7)
    with open_rollout_server(config, policy, inference_gpu=2) as active_server:
        assert active_server is server
        assert events == ["start", "init_weight_sync", "sync_policy_weights"]
        events.append("body")

    factory.assert_called_once_with(model_id="local/model", gpu=2, seed=7)
    server.init_weight_sync.assert_called_once_with("cuda:0")
    server.sync_policy_weights.assert_called_once_with(policy)
    assert events == ["start", "init_weight_sync", "sync_policy_weights", "body", "stop"]


@pytest.mark.parametrize("stage", ["start", "init_weight_sync", "sync_policy_weights", "body"])
def test_failure_is_propagated_and_server_is_stopped(policy, server_factory, stage):
    _, server = server_factory
    error = RuntimeError(f"failed at {stage}")
    if stage != "body":
        getattr(server, stage).side_effect = error

    with pytest.raises(RuntimeError) as caught:
        with open_rollout_server(GRPOExperimentConfig(), policy):
            if stage == "body":
                raise error
            pytest.fail("An initialization failure must prevent entry into the body.")

    assert caught.value is error
    server.stop.assert_called_once_with()
    if stage == "start":
        server.init_weight_sync.assert_not_called()
    if stage in ("start", "init_weight_sync"):
        server.sync_policy_weights.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_rejects_invalid_device_assignment_before_starting(policy, server_factory, device):
    factory, server = server_factory
    next(policy.parameters()).device = torch.device(device)
    with pytest.raises(ValueError):
        with open_rollout_server(GRPOExperimentConfig(), policy):
            pytest.fail("Invalid devices must not start a server.")
    factory.assert_not_called()
    server.start.assert_not_called()
