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
@pytest.mark.parametrize("normalizer", ["std", "mean"])
def test_tied_and_singleton_groups_produce_finite_zero_advantages(group_size, rewards, normalizer):
    advantages, _ = compute_group_normalized_rewards(
        torch.tensor(rewards), group_size, advantage_normalizer=normalizer
    )
    torch.testing.assert_close(advantages, torch.zeros(len(rewards)))
    assert torch.isfinite(advantages).all()


def test_adds_epsilon_to_the_standard_deviation():
    advantages, _ = compute_group_normalized_rewards(torch.tensor([1.0, 0.0]), 2, advantage_eps=1.0)
    expected = 0.5 / (2**-0.5 + 1.0)
    torch.testing.assert_close(advantages, torch.tensor([expected, -expected]))


def test_adds_epsilon_to_the_group_mean():
    advantages, _ = compute_group_normalized_rewards(
        torch.tensor([1.0, 0.0]), 2, advantage_eps=1.0, advantage_normalizer="mean"
    )
    torch.testing.assert_close(advantages, torch.tensor([1 / 3, -1 / 3]))


def test_mean_normalization_scales_sparse_successes_by_each_group_mean():
    advantages, _ = compute_group_normalized_rewards(
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
        4, advantage_normalizer="mean",
    )
    expected = torch.tensor([0.75, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, -0.75])
    expected = expected / torch.tensor([0.25 + 1e-6] * 4 + [0.75 + 1e-6] * 4)
    torch.testing.assert_close(advantages, expected)


@pytest.mark.parametrize(
    "rewards, group_size, eps",
    [([], 2, 1e-6), ([1.0, 0.0], 0, 1e-6), ([1.0, 0.0, 1.0], 2, 1e-6),
     ([[1.0, 0.0]], 2, 1e-6), ([1.0, 0.0], 2, 0.0)],
)
def test_rejects_invalid_groups(rewards, group_size, eps):
    with pytest.raises(ValueError):
        compute_group_normalized_rewards(torch.tensor(rewards), group_size, advantage_eps=eps)


@pytest.mark.parametrize("baseline", ["mean", "none"])
@pytest.mark.parametrize("normalizer", ["std", "none", "mean"])
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
    elif normalizer == "mean":
        expected = expected / (torch.tensor([0.5, 0.5, 3.0, 3.0]) + eps)
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


@pytest.mark.parametrize("normalizer", ["none", "mean"])
def test_normalization_supports_noncontiguous_integer_rewards(normalizer):
    rewards = torch.tensor([1, 9, 0, 9, 4, 9, 2, 9])[::2]
    original = rewards.clone()
    advantages, _ = compute_group_normalized_rewards(rewards, 2, advantage_normalizer=normalizer)
    assert not rewards.is_contiguous()
    assert advantages.dtype == torch.float32
    expected = torch.tensor([0.5, -0.5, 1.0, -1.0])
    if normalizer == "mean":
        expected = expected / (torch.tensor([0.5, 0.5, 3.0, 3.0]) + 1e-6)
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(rewards, original)


@pytest.mark.parametrize("eps", [0.0, float("nan"), float("inf"), -1.0])
@pytest.mark.parametrize("normalizer", ["std", "mean"])
def test_normalization_rejects_invalid_epsilon(eps, normalizer):
    with pytest.raises(ValueError, match="finite and positive"):
        compute_group_normalized_rewards(
            torch.tensor([1.0, 0.0]), 2, advantage_eps=eps, advantage_normalizer=normalizer
        )


def test_invalid_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline"):
        compute_group_normalized_rewards(torch.tensor([1.0, 0.0]), 2, baseline="invalid")


@pytest.mark.parametrize("baseline", ["mean", "none"])
def test_mean_normalization_handles_zero_rewards_without_nan(baseline):
    advantages, _ = compute_group_normalized_rewards(
        torch.zeros(4), 2, baseline=baseline, advantage_normalizer="mean"
    )
    torch.testing.assert_close(advantages, torch.zeros(4))
    assert torch.isfinite(advantages).all()


def test_invalid_normalizer_is_rejected():
    with pytest.raises(NotImplementedError):
        compute_group_normalized_rewards(
            torch.tensor([1.0, 0.0]), 2, advantage_normalizer="invalid"
        )
