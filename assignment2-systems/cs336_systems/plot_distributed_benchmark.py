"""Generate reproducible figures from the single-node All-Reduce benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot single-node All-Reduce benchmark results.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("results/distributed/single_node/all_reduce_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/distributed/single_node/figures"),
    )
    return parser.parse_args()


def load_successful_results(input_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    successful = [record for record in payload["results"] if record["status"] == "ok"]
    if not successful:
        raise ValueError("input contains no successful benchmark records")
    return pd.DataFrame(successful), payload["metadata"]


def _plot_metric(
    results: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_y: bool,
) -> None:
    figure, axes = plt.subplots(figsize=(8, 5))
    for world_size, group in results.groupby("world_size", sort=True):
        ordered = group.sort_values("size_mib")
        axes.plot(
            ordered["size_mib"],
            ordered[metric],
            marker="o",
            linewidth=2,
            label=f"{world_size} GPUs",
        )

    axes.set_xscale("log")
    if log_y:
        axes.set_yscale("log")
    axes.set_xlabel("Tensor size (MiB, float32)")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, which="both", linestyle="--", alpha=0.35)
    axes.legend(title="World size")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_figures(input_path: Path, output_dir: Path) -> list[Path]:
    results, metadata = load_successful_results(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = str(metadata["backend"]).upper()

    latency_path = output_dir / "all_reduce_latency.png"
    _plot_metric(
        results,
        metric="median_ms",
        ylabel="Median latency (ms)",
        title=f"Single-node All-Reduce latency ({backend})",
        output_path=latency_path,
        log_y=True,
    )

    bandwidth_path = output_dir / "all_reduce_bus_bandwidth.png"
    _plot_metric(
        results,
        metric="bus_bandwidth_gbps",
        ylabel="Estimated bus bandwidth (GB/s)",
        title=f"Single-node All-Reduce bandwidth ({backend})",
        output_path=bandwidth_path,
        log_y=False,
    )
    return [latency_path, bandwidth_path]


def main() -> None:
    args = parse_args()
    output_paths = generate_figures(args.input_path, args.output_dir)
    for output_path in output_paths:
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
