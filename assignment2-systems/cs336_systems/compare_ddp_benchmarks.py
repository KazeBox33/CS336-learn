"""Compare the educational DDP synchronization strategies."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cs336_systems.benchmark import MODEL_CONFIGS


AVAILABLE_IMPLEMENTATIONS = ("naive", "flat", "overlap")
DEFAULT_IMPLEMENTATIONS = ("naive", "flat")
IMPLEMENTATION_LABELS = {
    "naive": "Per-parameter",
    "flat": "Flat-gradient",
    "overlap": "Overlapped per-parameter",
}
COMPARABLE_FIELDS = (
    "backend",
    "model_size",
    "parameter_count",
    "world_size",
    "global_batch_size",
    "local_batch_size",
    "context_length",
    "vocab_size",
    "warmup_steps",
    "measurement_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare educational DDP training implementations."
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=AVAILABLE_IMPLEMENTATIONS,
        default=list(DEFAULT_IMPLEMENTATIONS),
        help="implementations to run sequentially; naive is the comparison baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _validate_comparable_results(results: dict[str, dict[str, Any]]) -> None:
    if "naive" not in results:
        raise ValueError("results must contain the naive baseline")
    if unknown := set(results) - set(AVAILABLE_IMPLEMENTATIONS):
        raise ValueError(f"unknown DDP implementations: {sorted(unknown)}")

    for implementation, result in results.items():
        if result.get("ddp_implementation") != implementation:
            raise ValueError(
                f"result labeled {implementation!r} contains "
                f"{result.get('ddp_implementation')!r}"
            )

    baseline = results["naive"]
    for implementation, result in results.items():
        if implementation == "naive":
            continue
        for field in COMPARABLE_FIELDS:
            if result.get(field) != baseline.get(field):
                raise ValueError(
                    f"benchmark results for {implementation!r} differ in {field!r}"
                )


def build_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build summary metrics from comparable benchmark result documents."""
    _validate_comparable_results(results)
    naive = results["naive"]
    naive_step_ms = float(naive["step_mean_ms"])
    if naive_step_ms <= 0:
        raise ValueError("mean step timing values must be positive")

    by_implementation: dict[str, dict[str, float]] = {}
    for implementation, result in results.items():
        if implementation == "naive":
            continue
        step_ms = float(result["step_mean_ms"])
        if step_ms <= 0:
            raise ValueError("mean step timing values must be positive")
        by_implementation[implementation] = {
            "step_speedup_vs_naive": naive_step_ms / step_ms,
            "step_time_reduction_pct_vs_naive": (
                naive_step_ms - step_ms
            )
            / naive_step_ms
            * 100.0,
        }

    comparison_metrics: dict[str, Any] = {
        "by_implementation": by_implementation,
    }
    if "flat" in results:
        flat = results["flat"]
        naive_communication_ms = float(naive["communication_mean_ms"])
        flat_communication_ms = float(flat["communication_mean_ms"])
        if min(naive_communication_ms, flat_communication_ms) <= 0:
            raise ValueError("mean communication timing values must be positive")

        # Keep the original flat-DDP fields for compatibility with existing results.
        comparison_metrics.update(
            {
                "step_speedup": by_implementation["flat"][
                    "step_speedup_vs_naive"
                ],
                "communication_speedup": naive_communication_ms
                / flat_communication_ms,
                "step_time_reduction_pct": by_implementation["flat"][
                    "step_time_reduction_pct_vs_naive"
                ],
                "communication_time_reduction_pct": (
                    naive_communication_ms - flat_communication_ms
                )
                / naive_communication_ms
                * 100.0,
            }
        )

    return {
        "benchmark": (
            "ddp_overlap_comparison"
            if "overlap" in results
            else "minimal_ddp_flat_comparison"
        ),
        "configuration": {field: naive[field] for field in COMPARABLE_FIELDS},
        "results": results,
        "comparison": comparison_metrics,
    }


def _run_benchmark(
    args: argparse.Namespace,
    implementation: str,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "cs336_systems.naive_ddp_benchmark",
        "--ddp-implementation",
        implementation,
        "--model-size",
        args.model_size,
        "--global-batch-size",
        str(args.global_batch_size),
        "--context-length",
        str(args.context_length),
        "--vocab-size",
        str(args.vocab_size),
        "--world-size",
        str(args.world_size),
        "--backend",
        args.backend,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--seed",
        str(args.seed),
        "--master-addr",
        args.master_addr,
        "--output-path",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _write_csv(comparison: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "implementation",
                "step_mean_ms",
                "step_std_ms",
                "post_backward_sync_or_wait_mean_ms",
                "post_backward_sync_or_wait_std_ms",
                "post_backward_sync_or_wait_fraction_pct",
                "timing_scope",
            ),
        )
        writer.writeheader()
        for implementation in comparison["results"]:
            result = comparison["results"][implementation]
            writer.writerow(
                {
                    "implementation": implementation,
                    "step_mean_ms": result["step_mean_ms"],
                    "step_std_ms": result["step_std_ms"],
                    "post_backward_sync_or_wait_mean_ms": result[
                        "communication_mean_ms"
                    ],
                    "post_backward_sync_or_wait_std_ms": result[
                        "communication_std_ms"
                    ],
                    "post_backward_sync_or_wait_fraction_pct": result[
                        "communication_fraction_pct"
                    ],
                    "timing_scope": result["communication_timing_scope"],
                }
            )


def format_markdown_report(comparison: dict[str, Any]) -> str:
    """Format the assignment table and commentary from structured results."""
    naive = comparison["results"]["naive"]
    metrics = comparison["comparison"]

    def describe_change(value: float) -> str:
        direction = "reduction" if value >= 0 else "increase"
        return f"{abs(value):.2f}% {direction}"

    title = (
        "# Overlapped DDP comparison"
        if "overlap" in comparison["results"]
        else "# Flat-gradient DDP comparison"
    )
    lines = [
        title,
        "",
        "| Implementation | Step mean (ms) | Step std (ms) | Post-backward sync/wait (ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for implementation, result in comparison["results"].items():
        lines.append(
            f"| {IMPLEMENTATION_LABELS[implementation]} | "
            f"{result['step_mean_ms']:.3f} | {result['step_std_ms']:.3f} | "
            f"{result['communication_mean_ms']:.3f} |"
        )
    lines.append("")

    if "flat" in comparison["results"]:
        flat = comparison["results"]["flat"]
        lines.append(
            "Flattening gradients into one collective changed mean communication "
            f"time from {naive['communication_mean_ms']:.3f} ms to "
            f"{flat['communication_mean_ms']:.3f} ms, a "
            f"{describe_change(metrics['communication_time_reduction_pct'])}."
        )
    if "overlap" in comparison["results"]:
        overlap = comparison["results"]["overlap"]
        overlap_metrics = metrics["by_implementation"]["overlap"]
        lines.append(
            f"Mean training-step time changed from {naive['step_mean_ms']:.3f} ms "
            f"with per-parameter synchronization to "
            f"{overlap['step_mean_ms']:.3f} ms with overlap "
            f"({overlap_metrics['step_speedup_vs_naive']:.3f}x relative speed)."
        )
    elif "flat" in comparison["results"]:
        flat = comparison["results"]["flat"]
        lines.append(
            f"Mean training-step time changed from {naive['step_mean_ms']:.3f} ms "
            f"to {flat['step_mean_ms']:.3f} ms "
            f"({metrics['step_speedup']:.3f}x relative speed)."
        )

    lines.append("")
    return "\n".join(lines)


def _print_summary(comparison: dict[str, Any]) -> None:
    print("implementation | step (ms) | post-backward sync/wait (ms) | scope")
    for implementation in comparison["results"]:
        result = comparison["results"][implementation]
        print(
            f"{implementation:>14} | "
            f"{result['step_mean_ms']:>9.3f} | "
            f"{result['communication_mean_ms']:>18.3f} | "
            f"{result['communication_timing_scope']}"
        )

    metrics = comparison["comparison"]
    for implementation, implementation_metrics in metrics[
        "by_implementation"
    ].items():
        print(
            f"{implementation} step speedup vs naive: "
            f"{implementation_metrics['step_speedup_vs_naive']:.3f}x"
        )


def main() -> None:
    args = parse_args()
    if len(args.implementations) != len(set(args.implementations)):
        raise ValueError("implementations must not contain duplicates")
    if "naive" not in args.implementations:
        raise ValueError("implementations must include naive as the baseline")

    output_dir = args.output_dir or Path(
        "results/distributed/overlap_ddp_comparison"
        if "overlap" in args.implementations
        else "results/distributed/flat_ddp_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        implementation: _run_benchmark(
            args,
            implementation,
            output_dir / f"{implementation}.json",
        )
        for implementation in args.implementations
    }
    comparison = build_comparison(results)

    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_csv(comparison, output_dir / "comparison.csv")
    (output_dir / "comparison.md").write_text(
        format_markdown_report(comparison),
        encoding="utf-8",
    )
    _print_summary(comparison)
    print(f"wrote {comparison_path}")


if __name__ == "__main__":
    main()
