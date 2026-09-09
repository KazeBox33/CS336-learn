import pytest

from cs336_systems.naive_ddp_benchmark import summarize_training_timings


def test_summarize_training_timings_uses_critical_rank() -> None:
    summary = summarize_training_timings(
        rank_step_timings_ms=[
            [10.0, 13.0, 11.0],
            [12.0, 11.0, 15.0],
        ],
        rank_communication_timings_ms=[
            [2.0, 4.0, 3.0],
            [3.0, 2.0, 5.0],
        ],
    )

    assert summary["step_mean_ms"] == pytest.approx((12.0 + 13.0 + 15.0) / 3.0)
    assert summary["communication_mean_ms"] == pytest.approx((3.0 + 4.0 + 5.0) / 3.0)
    assert summary["communication_fraction_pct"] == pytest.approx(30.0)


def test_summarize_training_timings_rejects_mismatched_measurements() -> None:
    with pytest.raises(ValueError, match="same non-zero number"):
        summarize_training_timings(
            rank_step_timings_ms=[[1.0, 2.0], [1.0]],
            rank_communication_timings_ms=[[0.5, 0.5], [0.5]],
        )
