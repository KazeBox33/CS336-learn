from unittest.mock import Mock, call

import pytest
import torch

from cs336_alignment.grpo_experiment import (
    GRPOExperimentConfig,
    GRPOExperimentData,
    evaluate_grpo_policy,
)
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


@pytest.fixture
def config():
    return GRPOExperimentConfig(n_val_examples=3, seed=7, sampling_max_tokens=64)


@pytest.fixture
def data():
    return GRPOExperimentData(
        train_examples=[], train_prompts=[],
        val_examples=[GSM8KExample(f"Q{i}", str(i)) for i in range(3)],
        val_prompts=[f"VAL_PROMPT_{i}" for i in range(3)],
    )


@pytest.fixture
def completions():
    return [
        VLLMCompletion("correct", [10, 11], "stop"),
        VLLMCompletion("wrong", [20, 21, 22, 23], "stop"),
        VLLMCompletion("", [], "length"),
    ]


def test_syncs_latest_weights_and_evaluates_one_response_per_question(config, data, completions):
    server = Mock(spec=VLLMServer)
    policy = Mock(spec=torch.nn.Module)
    events = []
    server.sync_policy_weights.side_effect = lambda model: events.append("sync")

    def generate(**kwargs):
        assert events == ["sync"]
        events.append("generate")
        return completions

    server.generate_completions.side_effect = generate
    reward_fn = Mock(side_effect=[
        {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0},
        {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0},
        {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0},
    ])
    metrics, returned = evaluate_grpo_policy(config, data, server, policy, reward_fn, batch_size=2)
    assert returned is completions
    assert events == ["sync", "generate"]
    server.sync_policy_weights.assert_called_once_with(policy)
    kwargs = server.generate_completions.call_args.kwargs
    assert kwargs["prompts"] is data.val_prompts
    assert kwargs["batch_size"] == 2
    assert kwargs["sampling_params"] == {
        "temperature": 1.0, "top_p": 1.0, "max_tokens": 64, "n": 1, "seed": 7,
        "stop": ["</answer>"], "include_stop_str_in_output": True,
    }
    assert reward_fn.call_args_list == [call("correct", "0"), call("wrong", "1"), call("", "2")]
    assert metrics == pytest.approx({
        "val_num_examples": 3,
        "val_mean_reward": 1 / 3,
        "val_mean_format_reward": 2 / 3,
        "val_accuracy": 1 / 3,
        "val_mean_response_length": 2.0,
        "val_length_limited_rate": 1 / 3,
    })


def test_validation_does_not_run_the_training_model_or_modify_gradients(
    config, data, completions, tiny_train_model
):
    policy = tiny_train_model
    original_mode = policy.training
    original = [p.detach().clone() for p in policy.parameters()]
    for param in policy.parameters():
        param.grad = torch.full_like(param, 0.125)
    policy.forward = Mock(side_effect=AssertionError("Validation should use vLLM, not policy.forward"))
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = completions
    reward_fn = Mock(return_value={"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0})
    evaluate_grpo_policy(config, data, server, policy, reward_fn)
    policy.forward.assert_not_called()
    assert policy.training == original_mode
    for param, initial in zip(policy.parameters(), original, strict=True):
        torch.testing.assert_close(param, initial)
        torch.testing.assert_close(param.grad, torch.full_like(param, 0.125))


@pytest.mark.parametrize("prompts", [[], ["VAL_PROMPT_0"]])
def test_invalid_validation_data_does_not_generate(config, data, prompts):
    invalid = GRPOExperimentData([], data.val_examples, [], prompts)
    server = Mock(spec=VLLMServer)
    with pytest.raises(ValueError, match="non-empty and aligned"):
        evaluate_grpo_policy(config, invalid, server, Mock(), Mock())
    server.sync_policy_weights.assert_not_called()
    server.generate_completions.assert_not_called()


def test_invalid_batch_size_does_not_generate(config, data):
    server = Mock(spec=VLLMServer)
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_grpo_policy(config, data, server, Mock(), Mock(), batch_size=0)
    server.sync_policy_weights.assert_not_called()


def test_response_count_mismatch_is_rejected_before_grading(config, data):
    server = Mock(spec=VLLMServer)
    server.generate_completions.return_value = []
    reward_fn = Mock()
    with pytest.raises(RuntimeError, match="Expected 3 validation responses"):
        evaluate_grpo_policy(config, data, server, Mock(), reward_fn)
    reward_fn.assert_not_called()


def test_sync_failure_prevents_validation_generation(config, data):
    server = Mock(spec=VLLMServer)
    server.sync_policy_weights.side_effect = RuntimeError("sync failed")
    with pytest.raises(RuntimeError, match="sync failed"):
        evaluate_grpo_policy(config, data, server, Mock(), Mock())
    server.generate_completions.assert_not_called()
