import random
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import vllm_utils
from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    GRPOExperimentData,
    generate_training_rollouts,
)
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


@pytest.fixture
def config():
    return GRPOExperimentConfig(
        rollout_batch_size=6, group_size=3, gradient_accumulation_steps=2,
        sampling_temperature=0.7, sampling_max_tokens=64,
    )


@pytest.fixture
def data():
    return GRPOExperimentData(
        train_examples=[GSM8KExample(f"Q{i}", str(i)) for i in range(4)],
        train_prompts=[f"PROMPT_{i}" for i in range(4)],
        val_examples=[],
        val_prompts=[],
    )


def test_generation_syncs_first_and_preserves_group_alignment(monkeypatch, config, data):
    policy = Mock(spec=torch.nn.Module)
    server = VLLMServer(model_id=config.model_id, launch_server=False)
    events = []
    requests = []

    def sync(model):
        assert model is policy
        events.append("sync")

    def request(method, url, payload, timeout):
        assert events == ["sync"]
        assert method == "POST"
        assert url.endswith("/v1/completions")
        requests.append(payload)
        events.append("generate")
        choices = [
            {
                "index": i * payload["n"] + j,
                "text": f"{prompt}:reply{j}",
                "token_ids": [i, j],
                "finish_reason": "stop",
            }
            for i, prompt in enumerate(payload["prompt"])
            for j in range(payload["n"])
        ]
        return {"choices": list(reversed(choices))}

    monkeypatch.setattr(server, "sync_policy_weights", sync)
    monkeypatch.setattr(vllm_utils, "_http_json", request)
    prompts, completions, truths = generate_training_rollouts(
        config, data, server, policy, rng=random.Random(7)
    )

    payload = requests[0]
    assert len(payload["prompt"]) == 2
    assert payload["n"] == 3
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 64
    assert payload["stop"] == ["</answer>"]
    assert payload["include_stop_str_in_output"] is True
    assert len(prompts) == len(completions) == len(truths) == 6
    for i, prompt in enumerate(payload["prompt"]):
        start = i * config.group_size
        assert prompts[start:start + 3] == [prompt] * 3
        assert truths[start:start + 3] == [prompt.removeprefix("PROMPT_")] * 3
        for j in range(3):
            assert completions[start + j] == VLLMCompletion(f"{prompt}:reply{j}", [i, j], "stop")
    assert events == ["sync", "generate"]


def test_sampling_rng_advances_and_same_seed_reproduces_batches(config, data):
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = [VLLMCompletion("reply", [], "stop")] * 6
    policy = Mock(spec=torch.nn.Module)

    def capture(seed):
        rng = random.Random(seed)
        result = []
        for _ in range(2):
            prompts, _, truths = generate_training_rollouts(config, data, server, policy, rng=rng)
            params = server.generate_completions.call_args.kwargs["sampling_params"]
            result.append((prompts, truths, params["seed"]))
        return result

    first = capture(7)
    assert first == capture(7)
    assert first != capture(8)
    assert first[0][2] != first[1][2]
    assert server.sync_policy_weights.call_count == server.generate_completions.call_count == 6


def test_incomplete_response_batch_is_rejected(config, data):
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = []
    with pytest.raises(RuntimeError, match="Expected 6 responses, received 0"):
        generate_training_rollouts(config, data, server, Mock(), rng=random.Random(0))


def test_sync_failure_prevents_generation(config, data):
    server = Mock(spec=VLLMServer)
    error = RuntimeError("weight sync failed")
    server.sync_policy_weights.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        generate_training_rollouts(config, data, server, Mock(), rng=random.Random(0))
    assert caught.value is error
    server.generate_completions.assert_not_called()


@pytest.mark.parametrize("prompts", [[], ["PROMPT_0"]])
def test_invalid_training_data_does_not_sync_or_generate(config, prompts):
    data = GRPOExperimentData(
        [GSM8KExample("Q0", "0"), GSM8KExample("Q1", "1")], [], prompts, []
    )
    server = Mock(spec=VLLMServer)
    with pytest.raises(ValueError, match="non-empty and aligned"):
        generate_training_rollouts(config, data, server, Mock(), rng=random.Random(0))
    server.sync_policy_weights.assert_not_called()
    server.generate_completions.assert_not_called()


@pytest.mark.parametrize("indices", [[0, 0, 2], [0, 1], [0, 1, 3]])
def test_completion_indices_cannot_silently_misalign_groups(monkeypatch, indices):
    monkeypatch.setattr(
        vllm_utils,
        "_http_json",
        lambda *_args, **_kwargs: {
            "choices": [{"index": i, "text": "reply"} for i in indices]
        },
    )
    with pytest.raises(RuntimeError, match="completion indices"):
        vllm_utils.generate_completions(
            "http://localhost", "MODEL", ["PROMPT"],
            {"temperature": 1.0, "max_tokens": 64, "n": 3, "seed": 0},
        )
