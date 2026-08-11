import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cs336_systems.benchmark import MODEL_CONFIGS


def load_results(input_dir: Path) -> pd.DataFrame:
    records = []

    for result_path in sorted(input_dir.glob("*.json")):  # 找到所有以.json结尾的文件 转化为对应的Path对象
        result = json.loads(result_path.read_text(encoding="utf-8"))
        records.append(result)

    return pd.DataFrame(records)


def build_phase_table(
    results: pd.DataFrame,
    batch_size: int,
    context_length: int,
    warmup_steps: int,
) -> pd.DataFrame:
    filtered = results[(results["batch_size"] == batch_size) & (results["context_length"] == context_length) & (results["warmup_steps"] == warmup_steps)]

    cumulative = filtered.pivot(  # 转换成新的表格
        index="model_size",
        columns="mode",
        values="mean_ms",
    )

    phases = pd.DataFrame(index=cumulative.index)  # 建立一张新表，并沿用模型名称作为行索引
    phases["forward"] = cumulative["forward"]
    phases["backward"] = cumulative["forward_backward"] - cumulative["forward"]
    phases["optimizer"] = cumulative["full"] - cumulative["forward_backward"]

    model_order = [model_size for model_size in MODEL_CONFIGS if model_size in phases.index]

    return phases.loc[model_order]


def build_precision_table(
    results: pd.DataFrame,
    batch_size: int,
    context_length: int,
    warmup_steps: int,
    mode: str,
) -> pd.DataFrame:
    filtered = results[
        (results["batch_size"] == batch_size)
        & (results["context_length"] == context_length)
        & (results["warmup_steps"] == warmup_steps)
        & (results["mode"] == mode)
    ]

    comparison = filtered.pivot(
        index="model_size",
        columns="compute_dtype",
        values="mean_ms",
    )
    comparison["speedup"] = comparison["float32"] / comparison["bfloat16"]

    model_order = [
        model_size
        for model_size in MODEL_CONFIGS
        if model_size in comparison.index
    ]

    return comparison.loc[model_order]


def plot_phase_times(
    phases: pd.DataFrame,
    output_path: Path,
    batch_size: int,
    context_length: int,
    warmup_steps: int,
) -> None:
    axes = phases.plot(
        kind="bar",
        figsize=(10, 6),
        color=["#2878B5", "#9AC9DB", "#F8AC8C"],
        rot=0,
    )

    axes.set_title(f"Transformer Training Step Breakdown\nbatch={batch_size}, context={context_length}, warmup={warmup_steps}")
    axes.set_xlabel("Model size")
    axes.set_ylabel("Mean time (ms)")
    axes.legend(title="Phase")
    axes.grid(axis="y", linestyle="--", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark results.")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/end_to_end"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/figures/phase_times.png"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = load_results(args.input_dir)
    phases = build_phase_table(
        results=results,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
    )

    print(phases)

    plot_phase_times(
        phases=phases,
        output_path=args.output_path,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
    )

    print(f"saved figure to {args.output_path}")


if __name__ == "__main__":
    main()
