import json
import math
import statistics
from pathlib import Path

TRAIN_METRICS = {
    "train_loss": "Policy gradient loss",
    "train_grad_norm": "Gradient norm (before clipping)",
    "train_mean_token_entropy": "Mean response token entropy",
    "train_mean_reward": "Train total reward",
    "train_mean_format_reward": "Train format reward",
}
VAL_METRICS = {
    "val_mean_reward": "Validation total reward",
    "val_mean_format_reward": "Validation format reward",
    "val_accuracy": "Validation accuracy",
    "val_mean_response_length": "Validation response length (tokens)",
}
METRICS = {**TRAIN_METRICS, **VAL_METRICS}
EXTRA_METRICS = {
    "train_clip_fraction": "Clipped fraction",
    "train_iteration_seconds": "Rollout and training iteration time (seconds)",
    "val_seconds": "Validation time (seconds)",
}
X_AXES = {
    "rollout_batches": "Completed rollout batches",
    "optimizer_updates": "Completed optimizer updates",
    "elapsed_seconds": "Elapsed loop time (seconds)",
}


def updates_per_rollout(config: dict) -> int:
    rollout = config.get("rollout_batch_size", 1)
    train = config.get("train_batch_size") or rollout
    if train <= 0 or rollout % train:
        raise ValueError("Invalid training batch size in run configuration.")
    return rollout // train


def summarize_values(values: list[float]) -> dict:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    margin = 1.96 * std / math.sqrt(len(values)) if std is not None else None
    return {
        "n": len(values), "mean": mean, "sample_std": std,
        "ci_lower": mean - margin if margin is not None else None,
        "ci_upper": mean + margin if margin is not None else None,
    }


def aggregate_grpo_runs(run_dirs: list[str | Path]) -> tuple[list[dict], dict]:
    """Validate comparable completed runs and align samples by rollout-batch index."""
    if not run_dirs:
        raise ValueError("Provide at least one completed run.")
    runs, seeds = [], set()
    reference = None
    runtime_reference = None
    for directory in run_dirs:
        root = Path(directory).resolve()
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if manifest.get("completed") is not True:
            raise ValueError(f"Run is incomplete: {root}")
        runtime = {name: manifest.get(name) for name in (
            "training_device", "inference_gpu", "validation_batch_size", "checkpoint_interval",
        )}
        if runtime_reference is None:
            runtime_reference = runtime
        elif runtime != runtime_reference:
            raise ValueError("Runs must share device, validation batch and checkpoint settings.")
        config = dict(manifest["config"])
        factor = updates_per_rollout(config)
        seed = config.pop("seed")
        if seed in seeds:
            raise ValueError(f"Duplicate random seed: {seed}")
        seeds.add(seed)
        if reference is None:
            reference = config
        elif config != reference:
            raise ValueError("Runs must share the same model, data paths and training configuration except seed.")
        records = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        expected_steps = list(range(config["num_rollout_steps"] + 1))
        if [record.get("step") for record in records] != expected_steps:
            raise ValueError(f"Run must contain exactly one record for each step, including step 0: {root}")
        for record in records:
            step = record["step"]
            required = set(TRAIN_METRICS) if step > 0 else set()
            if step == 0 or step % config["eval_interval"] == 0 or step == config["num_rollout_steps"]:
                required.update(VAL_METRICS)
            if not required.issubset(record):
                raise ValueError(f"Missing metrics at step {step}: {sorted(required - record.keys())}")
            if step > 0 and config.get("importance_reweighting_method") in ("grpo", "gspo"):
                if "train_clip_fraction" not in record:
                    raise ValueError(f"Missing clip fraction at step {step}: {root}")
            for key, value in record.items():
                if key != "step" and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    raise ValueError(f"Metric {key} must be a finite number.")
            if "val_accuracy" in record and not 0 <= record["val_accuracy"] <= 1:
                raise ValueError("Validation accuracy must be between 0 and 1.")
            if "train_clip_fraction" in record and not 0 <= record["train_clip_fraction"] <= 1:
                raise ValueError("Clip fraction must be between 0 and 1.")
            if "optimizer_updates" in record and record["optimizer_updates"] != step * factor:
                raise ValueError("Optimizer update count does not match the run configuration.")
            for name in ("elapsed_seconds", "train_iteration_seconds", "val_seconds"):
                if name in record and record[name] < 0:
                    raise ValueError(f"{name} must be non-negative.")
        elapsed = [record["elapsed_seconds"] for record in records if "elapsed_seconds" in record]
        if elapsed and (len(elapsed) != len(records) or elapsed != sorted(elapsed)):
            raise ValueError("Elapsed time must be present at every step and nondecreasing.")
        runs.append({"path": str(root), "seed": seed, "records": records})

    curves = {}
    metric_titles = {**METRICS, **{
        name: title for name, title in EXTRA_METRICS.items()
        if any(name in record for run in runs for record in run["records"])
    }}
    for name in metric_titles:
        by_step = {}
        elapsed_by_step = {}
        for run in runs:
            for record in run["records"]:
                if name in record:
                    by_step.setdefault(record["step"], []).append(record[name])
                    if "elapsed_seconds" in record:
                        elapsed_by_step.setdefault(record["step"], []).append(record["elapsed_seconds"])
        curves[name] = []
        for step, values in sorted(by_step.items()):
            point = {"step": step, "optimizer_updates": step * factor, **summarize_values(values)}
            times = elapsed_by_step.get(step, [])
            if len(times) == len(values):
                point["elapsed_seconds"] = statistics.mean(times)
            curves[name].append(point)
    final = summarize_values([run["records"][-1]["val_accuracy"] for run in runs])
    summary = {
        "num_runs": len(runs), "seeds": [run["seed"] for run in runs],
        "run_paths": [run["path"] for run in runs],
        "config": reference, "final_val_accuracy": final,
        "runtime_settings": runtime_reference,
        "four_seed_requirement_met": len(runs) == 4,
        "meets_25_percent_target": final["mean"] >= 0.25 if len(runs) == 4 else None,
        "interval_method": "Normal approximation: mean +/- 1.96 * sample_std / sqrt(n); not an exact small-sample interval.",
        "metrics": curves,
        "metric_titles": metric_titles,
        "optimizer_steps_per_rollout": factor,
        "clip_fraction_unit": "response" if reference.get("importance_reweighting_method") == "gspo" else "token",
    }
    return runs, summary


def _plot_metric(groups: dict, name: str, title: str, output_path: Path, x_axis: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_axis not in X_AXES:
        raise ValueError(f"Unknown plot axis: {x_axis}")

    def x_value(point, factor):
        if x_axis == "rollout_batches":
            return point["step"]
        if x_axis == "optimizer_updates":
            return point["step"] * factor
        if "elapsed_seconds" not in point:
            raise ValueError("Elapsed-time plots require timing logs for every run.")
        return point["elapsed_seconds"]

    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        for index, (label, (runs, summary)) in enumerate(groups.items()):
            factor = summary["optimizer_steps_per_rollout"]
            color = f"C{index % 10}"
            points = summary["metrics"].get(name, [])
            if not points:
                continue
            legend = label
            if name == "train_clip_fraction":
                legend += f" ({summary['clip_fraction_unit']})"
            for run in runs:
                samples = [record for record in run["records"] if name in record]
                ax.plot([x_value(point, factor) for point in samples], [point[name] for point in samples],
                        color=color, alpha=0.2, linewidth=1)
            ax.plot([x_value(point, factor) for point in points], [point["mean"] for point in points],
                    color=color, linewidth=2, label=legend)
            band = [point for point in points if point["ci_lower"] is not None]
            if band:
                ax.fill_between([x_value(point, factor) for point in band],
                                [point["ci_lower"] for point in band], [point["ci_upper"] for point in band],
                                color=color, alpha=0.15)
        ax.set(title=title, xlabel=X_AXES[x_axis], ylabel=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)


def _validate_plot_axis(groups: dict, x_axis: str) -> None:
    if x_axis not in X_AXES:
        raise ValueError(f"Unknown plot axis: {x_axis}")
    if x_axis == "elapsed_seconds" and any(
        "elapsed_seconds" not in record
        for runs, _ in groups.values() for run in runs for record in run["records"]
    ):
        raise ValueError("Elapsed-time plots require timing logs for every run.")


def write_grpo_report(
    run_dirs: list[str | Path], output_dir: str | Path, *, x_axis: str = "rollout_batches",
) -> Path:
    """Plot real logged metrics and create a report scaffold without inventing conclusions."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Report directory already exists: {output_dir}")
    runs, summary = aggregate_grpo_runs(run_dirs)
    _validate_plot_axis({"mean": (runs, summary)}, x_axis)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, title in summary["metric_titles"].items():
        _plot_metric({"mean": (runs, summary)}, name, title, output_dir / f"{name}.png", x_axis)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    lines = [
        "# GRPO Experiment", "",
        f"Model: `{summary['config']['model_id']}`. Seeds: {summary['seeds']}.",
        f"Final mean validation accuracy: {summary['final_val_accuracy']['mean']:.2%}.",
        f"Four-seed requirement met: {summary['four_seed_requirement_met']}.",
        f"25% target met: {summary['meets_25_percent_target']} (null means fewer/more than four seeds).",
        "", "Only completed, comparable runs are included. No conclusions are inferred from reward alone.",
        "Shading uses a normal approximation, not an exact confidence interval with four seeds.",
        f"Plot axis: {X_AXES[x_axis]}. Each step in the log is a rollout batch.",
        "Elapsed-time mean curves pair mean reward with mean time at each rollout step; no interpolation is used.",
        f"Clip fraction counts active {summary['clip_fraction_unit']} units; zero-advantage rows are excluded.",
        "", "## Curves", "",
    ]
    for name, title in summary["metric_titles"].items():
        lines.extend([f"### {title}", f"![{title}]({name}.png)",
                      "Analysis pending: describe the trend and variation across seeds.", ""])
    lines.extend(["## Response Inspection (Pending)", "",
                  "Inspect each run's rollouts/val_step_0000.jsonl and final val_step file.",
                  "Compare the same response_index across validation snapshots: prompts remain fixed.",
                  "Include a few actual prompt/response examples before and after training.",
                  "Training snapshots use pre-update weights; validation snapshots use post-update weights.",
                  "", "## Debugging Check (Pending)", "",
                  "For the separate roughly-50-step smoke run, inspect reward changes and sensible rollouts.",
                  "Do not count a 50-step smoke run as one of the four full-length runs.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir / "report.md"


def write_grpo_comparison(
    run_dirs: list[str | Path], output_dir: str | Path, *, x_axis: str = "rollout_batches",
) -> Path:
    """Group methods, aggregate seeds within each method, then compare real logs."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Report directory already exists: {output_dir}")
    grouped, reference = {}, None
    for directory in run_dirs:
        root = Path(directory).resolve()
        config = json.loads((root / "run.json").read_text(encoding="utf-8"))["config"]
        method = config.get("importance_reweighting_method", "none")
        factor = updates_per_rollout(config)
        if factor == 1 and method == "none":
            label = "onpolicy"
        else:
            names = {"none": "naive", "noclip": "noclip", "grpo": "clip", "gspo": "gspo"}
            if factor == 1 or method not in names:
                raise ValueError("Comparison expects on-policy baseline or off-policy methods.")
            label = "offpolicy_" + names[method]
        common = {key: value for key, value in config.items() if key not in {
            "seed", "train_batch_size", "gradient_accumulation_steps", "importance_reweighting_method", "cliprange",
        }}
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError("Methods must keep model, data and other hyperparameters fixed.")
        grouped.setdefault(label, []).append(root)
    if len(grouped) < 2:
        raise ValueError("Provide at least two different methods for comparison.")
    groups = {label: aggregate_grpo_runs(roots) for label, roots in sorted(grouped.items())}
    _validate_plot_axis(groups, x_axis)
    settings = [summary["runtime_settings"] for _, summary in groups.values()]
    if any(runtime != settings[0] for runtime in settings):
        raise ValueError("Methods must share device, validation batch and checkpoint settings.")
    factors = {summary["optimizer_steps_per_rollout"] for label, (_, summary) in groups.items() if label != "onpolicy"}
    if len(factors) != 1:
        raise ValueError("Off-policy methods must use the same number of updates per rollout.")
    titles = {name: title for _, summary in groups.values() for name, title in summary["metric_titles"].items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, title in titles.items():
        _plot_metric(groups, name, title, output_dir / f"{name}.png", x_axis)
    required = {"offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo"}
    complete = required.issubset(groups) and all(groups[label][1]["four_seed_requirement_met"] for label in required)
    summary = {
        "x_axis": x_axis, "methods": {label: stats for label, (_, stats) in groups.items()},
        "four_method_four_seed_requirement_met": complete,
        "onpolicy_baseline_included": "onpolicy" in groups,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# On/Off-Policy GRPO Comparison", "",
             f"Plot axis: {X_AXES[x_axis]}.",
             f"Four methods with four seeds each: {complete}. On-policy baseline included: {'onpolicy' in groups}.",
             "Thin lines show individual seeds; bands are normal-approximation 95% mean intervals.",
             "Elapsed-time means pair mean time with mean reward at the same rollout step; no interpolation is used.",
             "GRPO clip fraction counts active response tokens; GSPO counts active responses. Zero-advantage rows are excluded.",
             "Loss values from different objectives are not directly comparable as model-quality scores.", "",
             "| Method | Seeds | Final Validation Accuracy (Mean +/- Sample Std) |", "| --- | --- | --- |"]
    for label, (_, stats) in groups.items():
        final = stats["final_val_accuracy"]
        std = "unavailable" if final["sample_std"] is None else f"{final['sample_std']:.2%}"
        lines.append(f"| {label} | {stats['seeds']} | {final['mean']:.2%} +/- {std} |")
    lines.extend(["", "Four seeds are required per method; smaller groups are previews, not completed experiments.", ""])
    for name, title in titles.items():
        lines.extend([f"## {title}", f"![{title}]({name}.png)",
                      "Analysis pending: compare performance, stability and between-seed variation.", ""])
    lines.extend(["## Response Inspection (Pending)",
                  "Inspect actual initial/final rollouts for each method before drawing conclusions.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir / "report.md"
