from __future__ import annotations

import pytest

from cs336_systems.compare_ddp_benchmarks import (
    build_comparison,
    format_markdown_report,
)


def _result(
    implementation: str,
    *,
    step_mean_ms: float,
    communication_mean_ms: float,
) -> dict[str, object]:
    return {
        "ddp_implementation": implementation,
        "backend": "nccl",
        "model_size": "xl",
        "parameter_count": 1_000,
        "world_size": 2,
        "global_batch_size": 4,
        "local_batch_size": 2,
        "context_length": 512,
        "vocab_size": 10_000,
        "warmup_steps": 5,
        "measurement_steps": 10,
        "step_mean_ms": step_mean_ms,
        "communication_mean_ms": communication_mean_ms,
    }


def test_build_comparison_computes_speedups() -> None:
    comparison = build_comparison(
        {
            "naive": _result(
                "naive",
                step_mean_ms=120.0,
                communication_mean_ms=40.0,
            ),
            "flat": _result(
                "flat",
                step_mean_ms=100.0,
                communication_mean_ms=10.0,
            ),
        }
    )

    metrics = comparison["comparison"]
    assert metrics["step_speedup"] == pytest.approx(1.2)
    assert metrics["communication_speedup"] == pytest.approx(4.0)
    assert metrics["step_time_reduction_pct"] == pytest.approx(100.0 / 6.0)
    assert metrics["communication_time_reduction_pct"] == pytest.approx(75.0)


def test_build_comparison_rejects_different_configurations() -> None:
    naive = _result("naive", step_mean_ms=120.0, communication_mean_ms=40.0)
    flat = _result("flat", step_mean_ms=100.0, communication_mean_ms=10.0)
    flat["world_size"] = 4

    with pytest.raises(ValueError, match="world_size"):
        build_comparison({"naive": naive, "flat": flat})


def test_format_markdown_report_uses_comparison_results() -> None:
    naive = _result("naive", step_mean_ms=120.0, communication_mean_ms=40.0)
    flat = _result("flat", step_mean_ms=100.0, communication_mean_ms=10.0)
    naive["communication_fraction_pct"] = 100.0 / 3.0
    flat["communication_fraction_pct"] = 10.0

    report = format_markdown_report(build_comparison({"naive": naive, "flat": flat}))

    assert "| Per-parameter | 120.000 | 40.000 | 33.33% |" in report
    assert "| Flat-gradient | 100.000 | 10.000 | 10.00% |" in report
    assert "75.00% reduction" in report
    assert "1.200x relative speed" in report
