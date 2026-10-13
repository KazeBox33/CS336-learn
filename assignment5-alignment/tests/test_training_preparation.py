from unittest.mock import Mock, call

import pytest
import torch

from cs336_alignment.training import _prepare_rollout_batch


def test_prepares_group_advantages_tokens_and_logging_metadata(tokenizer):
    prompts = ["Hello", "Hello", "This is", "This is"]
    responses = ["world", "test", "a test", "another test"]
    ground_truths = ["42"] * 4
    reward_fn = Mock(
        side_effect=[
            {"reward": r, "format_reward": 1.0, "answer_reward": r}
            for r in [1.0, 0.0, 0.5, 0.0]
        ]
    )
    token_batch, advantages, metadata = _prepare_rollout_batch(
        tokenizer, reward_fn, prompts, responses, ground_truths, group_size=2
    )
    assert reward_fn.call_args_list == [call(response, "42") for response in responses]
    assert token_batch["input_ids"].tolist() == [[3, 4, 0], [3, 8, 0], [5, 6, 7], [5, 6, 9]]
    assert token_batch["labels"].tolist() == [[4, 0, 0], [8, 0, 0], [6, 7, 8], [6, 9, 8]]
    assert token_batch["response_mask"].tolist() == [
        [True, False, False], [True, False, False], [False, True, True], [False, True, True]
    ]
    eps = 1e-6
    expected = torch.tensor([
        0.5 / (2**-0.5 + eps), -0.5 / (2**-0.5 + eps),
        0.25 / (0.5 * 2**-0.5 + eps), -0.25 / (0.5 * 2**-0.5 + eps),
    ])
    torch.testing.assert_close(advantages, expected)
    assert advantages.shape == (4,)
    assert not advantages.requires_grad
    assert all(tensor.device.type == "cpu" for tensor in [*token_batch.values(), advantages])
    assert metadata["mean_reward"] == 0.375
    assert metadata["mean_format_reward"] == 1.0
    assert metadata["mean_group_std"] == pytest.approx(0.75 * 2**-0.5)


def test_forwards_the_advantage_epsilon(tokenizer):
    reward_fn = Mock(side_effect=[
        {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0},
        {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0},
    ])
    _, advantages, _ = _prepare_rollout_batch(
        tokenizer, reward_fn, ["Hello"] * 2, ["world", "test"], ["42"] * 2,
        group_size=2, advantage_eps=1.0,
    )
    value = 0.5 / (2**-0.5 + 1.0)
    torch.testing.assert_close(advantages, torch.tensor([value, -value]))


@pytest.mark.parametrize(
    "prompts, responses, ground_truths, group_size",
    [
        ([], [], [], 2),
        (["Hello"], ["world", "test"], ["42"] * 2, 2),
        (["Hello"] * 2, ["world", "test"], ["42"], 2),
        (["Hello"] * 3, ["world"] * 3, ["42"] * 3, 2),
        (["Hello"] * 2, ["world"] * 2, ["42"] * 2, 0),
    ],
)
def test_rejects_unaligned_batches_before_scoring(tokenizer, prompts, responses, ground_truths, group_size):
    reward_fn = Mock()
    with pytest.raises(ValueError):
        _prepare_rollout_batch(tokenizer, reward_fn, prompts, responses, ground_truths, group_size)
    reward_fn.assert_not_called()
