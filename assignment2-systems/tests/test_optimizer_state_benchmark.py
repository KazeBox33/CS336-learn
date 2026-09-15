import pytest

from cs336_systems.optimizer_state_benchmark import (
    render_report,
    summarize_rank_timings,
    theoretical_fp32_memory_bytes,
)


def test_theoretical_fp32_memory_accounts_for_sharded_adam_state() -> None:
    baseline = theoretical_fp32_memory_bytes(100, world_size=2, sharded=False)
    sharded = theoretical_fp32_memory_bytes(100, world_size=2, sharded=True)

    assert baseline == {
        "parameters": 400.0,
        "gradients": 400.0,
        "optimizer_state": 800.0,
        "total": 1600.0,
    }
    assert sharded == {
        "parameters": 400.0,
        "gradients": 400.0,
        "optimizer_state": 400.0,
        "total": 1200.0,
    }


def test_summarize_rank_timings_uses_iteration_critical_path() -> None:
    summary = summarize_rank_timings([[1.0, 4.0], [3.0, 2.0]])

    assert summary["step_mean_ms"] == 3.5
    assert summary["step_std_ms"] == pytest.approx(0.70710678)


def test_render_report_uses_structured_results() -> None:
    checkpoint = {
        "allocated_bytes": 1024**2,
        "reserved_bytes": 1024**2,
        "phase_peak_allocated_bytes": 2 * 1024**2,
        "phase_peak_reserved_bytes": 2 * 1024**2,
    }
    result = {
        "maximum_memory_across_ranks": {
            "after_model_initialization": checkpoint,
            "before_optimizer_step": checkpoint,
            "after_optimizer_step": checkpoint,
        },
        "maximum_component_bytes_across_ranks": {
            "parameters": 4 * 1024**2,
            "gradients": 4 * 1024**2,
            "local_optimizer_state": 8 * 1024**2,
        },
        "theoretical_fp32_bytes_per_rank": {"optimizer_state": 1024**2},
        "step_mean_ms": 2.5,
        "step_std_ms": 0.25,
    }

    report = render_report({"results": {"baseline": result, "sharded": result}})

    assert "| baseline | 2.0 | 2.0 | 2.0 | 1.0 |" in report
    assert "| baseline | 4.0 | 4.0 | 8.0 |" in report
    assert "| sharded | 2.500 | 0.250 |" in report
    assert "Mean iteration time changed from 2.500 ms to 2.500 ms (+0.0%)." in report
