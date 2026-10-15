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
    "baseline, normalizer", [("mean", "std"), ("mean", "none"), ("none", "std"), ("none", "none")]
)
def test_baseline_and_normalizer_are_independent_and_preserve_inputs(baseline, normalizer):
    rewards = torch.tensor([1.0, 0.0, 4.0, 2.0])
    original = rewards.clone()
    eps = 1e-6
    advantages, metadata = compute_group_normalized_rewards(
        rewards, 2, baseline=baseline, advantage_normalizer=normalizer, advantage_eps=eps
    )
    expected = original.clone()
    if baseline == "mean":
        expected = expected - torch.tensor([0.5, 0.5, 3.0, 3.0])
    if normalizer == "std":
        expected = expected / (torch.tensor([2**-0.5, 2**-0.5, 2**0.5, 2**0.5]) + eps)
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(rewards, original)
    assert advantages.shape == rewards.shape
    assert advantages.device == rewards.device
    assert metadata["mean_reward"] == 1.75
    assert metadata["mean_group_std"] == pytest.approx((2**-0.5 + 2**0.5) / 2)


@pytest.mark.parametrize("baseline", ["mean", "none"])
@pytest.mark.parametrize("group_size", [1, 2])
def test_no_normalization_handles_tied_and_singleton_groups(baseline, group_size):
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    advantages, metadata = compute_group_normalized_rewards(
        rewards, group_size, baseline=baseline, advantage_normalizer="none", advantage_eps=0.0
    )
    expected = torch.zeros_like(rewards) if baseline == "mean" else rewards
    torch.testing.assert_close(advantages, expected)
    assert torch.isfinite(advantages).all()
    assert metadata["mean_group_std"] == 0.0


@pytest.mark.parametrize("eps", [0.0, 1.0, float("nan")])
def test_unnormalized_advantages_do_not_depend_on_epsilon(eps):
    advantages, _ = compute_group_normalized_rewards(
        torch.tensor([1.0, 0.0, 0.0, 1.0]), 2,
        advantage_normalizer="none", advantage_eps=eps,
    )
    torch.testing.assert_close(advantages, torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_no_normalization_supports_noncontiguous_integer_rewards():
    rewards = torch.tensor([1, 9, 0, 9, 4, 9, 2, 9])[::2]
    original = rewards.clone()
    advantages, _ = compute_group_normalized_rewards(rewards, 2, advantage_normalizer="none")
    assert not rewards.is_contiguous()
    assert advantages.dtype == torch.float32
    torch.testing.assert_close(advantages, torch.tensor([0.5, -0.5, 1.0, -1.0]))
    torch.testing.assert_close(rewards, original)


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), -1.0])
def test_std_normalization_rejects_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="finite and positive"):
        compute_group_normalized_rewards(torch.tensor([1.0, 0.0]), 2, advantage_eps=eps)


def test_invalid_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline"):
        compute_group_normalized_rewards(torch.tensor([1.0, 0.0]), 2, baseline="invalid")


def test_mean_normalization_is_still_explicitly_unsupported():
    with pytest.raises(NotImplementedError):
        compute_group_normalized_rewards(
            torch.tensor([1.0, 0.0]), 2, advantage_normalizer="mean"
        )
