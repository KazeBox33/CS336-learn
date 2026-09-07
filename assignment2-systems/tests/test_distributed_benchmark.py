import pytest

from cs336_systems.distributed_benchmark import mib_to_numel, summarize_rank_timings


def test_mib_to_numel_for_float32() -> None:
    assert mib_to_numel(1.0) == 262_144
    assert mib_to_numel(1024.0) == 268_435_456


@pytest.mark.parametrize("size_mib", [0.0, -1.0])
def test_mib_to_numel_rejects_non_positive_sizes(size_mib: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        mib_to_numel(size_mib)


def test_summarize_rank_timings_uses_slowest_rank_per_repetition() -> None:
    summary = summarize_rank_timings(
        [
            [1.0, 4.0, 2.0],
            [2.0, 3.0, 5.0],
        ],
        tensor_bytes=1_000_000,
        world_size=2,
    )

    # The critical-path samples are max([1, 2]), max([4, 3]), max([2, 5]).
    assert summary["mean_ms"] == pytest.approx((2.0 + 4.0 + 5.0) / 3.0)
    assert summary["median_ms"] == pytest.approx(4.0)
    assert summary["algorithmic_bandwidth_gbps"] == pytest.approx(0.25)
    assert summary["bus_bandwidth_gbps"] == pytest.approx(0.25)
