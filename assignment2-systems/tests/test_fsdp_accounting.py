import pytest

from cs336_systems.fsdp_accounting import (
    summarize_rank_timings,
    theoretical_fsdp_memory_bytes,
)


def test_theoretical_fsdp_memory_accounts_for_replicated_parameters() -> None:
    accounting = theoretical_fsdp_memory_bytes(
        80,
        20,
        world_size=2,
    )

    assert accounting == {
        "baseline_bytes_per_rank": 1600.0,
        "fsdp_bytes_per_rank": 960.0,
        "saved_bytes_per_rank": 640.0,
        "saved_fraction": 0.4,
    }


@pytest.mark.parametrize(
    ("sharded", "replicated", "world_size"),
    [(-1, 0, 2), (0, -1, 2), (0, 0, 0)],
)
def test_theoretical_fsdp_memory_rejects_invalid_inputs(
    sharded: int,
    replicated: int,
    world_size: int,
) -> None:
    with pytest.raises(ValueError):
        theoretical_fsdp_memory_bytes(
            sharded,
            replicated,
            world_size=world_size,
        )


def test_summarize_rank_timings_uses_iteration_critical_path() -> None:
    summary = summarize_rank_timings(
        [[1.0, 4.0], [3.0, 2.0]],
        [[10.0, 13.0], [12.0, 11.0]],
    )

    assert summary == {
        "forward_mean_ms": 3.5,
        "forward_std_ms": pytest.approx(0.70710678),
        "step_mean_ms": 12.5,
        "step_std_ms": pytest.approx(0.70710678),
    }


def test_summarize_rank_timings_rejects_mismatched_measurements() -> None:
    with pytest.raises(ValueError, match="same non-zero number"):
        summarize_rank_timings([[1.0], [2.0]], [[3.0], []])
