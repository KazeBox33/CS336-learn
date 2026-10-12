from unittest.mock import Mock, call

import pytest
import torch

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.rewards import compute_rollout_rewards


def test_scores_pairs_in_order_and_reports_batch_means():
    reward_fn = Mock(
        side_effect=[
            {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0},
            {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0},
            {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0},
            {"reward": 0.5, "format_reward": 1.0, "answer_reward": 0.5},
        ]
    )
    rewards, metadata = compute_rollout_rewards(
        reward_fn, ["a", "b", "c", "d"], ["2", "2", "5", "5"]
    )
    assert reward_fn.call_args_list == [call("a", "2"), call("b", "2"), call("c", "5"), call("d", "5")]
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0, 0.5]))
    assert rewards.shape == (4,)
    assert rewards.dtype == torch.float32
    assert not rewards.requires_grad
    assert metadata == {"mean_reward": 0.375, "mean_format_reward": 0.75}
    assert all(isinstance(value, float) for value in metadata.values())


def test_format_reward_is_not_added_to_total_reward():
    reward_fn = Mock(return_value={"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0})
    rewards, metadata = compute_rollout_rewards(reward_fn, ["wrong answer"], ["2"])
    torch.testing.assert_close(rewards, torch.zeros(1))
    assert metadata == {"mean_reward": 0.0, "mean_format_reward": 1.0}


@pytest.mark.parametrize("responses, truths", [([], []), (["a"], []), ([], ["2"])])
def test_rejects_empty_or_mismatched_batches_before_scoring(responses, truths):
    reward_fn = Mock()
    with pytest.raises(ValueError):
        compute_rollout_rewards(reward_fn, responses, truths)
    reward_fn.assert_not_called()


def test_uses_the_provided_math_grader():
    responses = [
        "1 + 1 = 2.</think> <answer>2</answer>",
        "1 + 1 = 3.</think> <answer>3</answer>",
        "2",
    ]
    rewards, metadata = compute_rollout_rewards(r1_zero_reward_fn, responses, ["2"] * 3)
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0]))
    assert metadata["mean_reward"] == pytest.approx(1 / 3)
    assert metadata["mean_format_reward"] == pytest.approx(2 / 3)
