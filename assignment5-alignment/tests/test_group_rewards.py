import pytest
import torch

from cs336_alignment.rewards import compute_group_normalized_rewards


def test_normalizes_each_group_independently_without_mutating_rewards():
    rewards = torch.tensor([1.0, 0.0, 4.0, 2.0])
    original = rewards.clone()
    eps = 1e-6
    advantages, metadata = compute_group_normalized_rewards(rewards, group_size=2)
    expected = torch.tensor(
        [0.5 / (2**-0.5 + eps), -0.5 / (2**-0.5 + eps),
         1.0 / (2**0.5 + eps), -1.0 / (2**0.5 + eps)]
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(rewards, original)
    assert advantages.shape == rewards.shape
    assert advantages.device == rewards.device
    assert metadata["mean_reward"] == 1.75
    assert metadata["mean_group_std"] == pytest.approx((2**-0.5 + 2**0.5) / 2)


@pytest.mark.parametrize("group_size, rewards", [(2, [1.0, 1.0, 0.0, 0.0]), (1, [1.0, 0.0])])
def test_tied_and_singleton_groups_produce_finite_zero_advantages(group_size, rewards):
    advantages, _ = compute_group_normalized_rewards(torch.tensor(rewards), group_size)
    torch.testing.assert_close(advantages, torch.zeros(len(rewards)))
    assert torch.isfinite(advantages).all()


def test_adds_epsilon_to_the_standard_deviation():
    advantages, _ = compute_group_normalized_rewards(torch.tensor([1.0, 0.0]), 2, advantage_eps=1.0)
    expected = 0.5 / (2**-0.5 + 1.0)
    torch.testing.assert_close(advantages, torch.tensor([expected, -expected]))


@pytest.mark.parametrize(
    "rewards, group_size, eps",
    [([], 2, 1e-6), ([1.0, 0.0], 0, 1e-6), ([1.0, 0.0, 1.0], 2, 1e-6),
     ([[1.0, 0.0]], 2, 1e-6), ([1.0, 0.0], 2, 0.0)],
)
def test_rejects_invalid_groups(rewards, group_size, eps):
    with pytest.raises(ValueError):
        compute_group_normalized_rewards(torch.tensor(rewards), group_size, advantage_eps=eps)


@pytest.mark.parametrize(
    "baseline, normalizer", [("none", "std"), ("mean", "none"), ("mean", "mean")]
)
def test_later_algorithm_variants_are_explicitly_unsupported(baseline, normalizer):
    with pytest.raises(NotImplementedError):
        compute_group_normalized_rewards(
            torch.tensor([1.0, 0.0]), 2, baseline=baseline, advantage_normalizer=normalizer
        )
