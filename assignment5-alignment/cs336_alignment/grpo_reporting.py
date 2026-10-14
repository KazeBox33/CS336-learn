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
    """Validate comparable completed runs and align metric samples by update step."""
    if not run_dirs:
        raise ValueError("Provide at least one completed run.")
    runs, seeds = [], set()
    reference = None
    for directory in run_dirs:
        root = Path(directory).resolve()
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if manifest.get("completed") is not True:
            raise ValueError(f"Run is incomplete: {root}")
        config = dict(manifest["config"])
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
            for key, value in record.items():
                if key != "step" and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    raise ValueError(f"Metric {key} must be a finite number.")
            if "val_accuracy" in record and not 0 <= record["val_accuracy"] <= 1:
                raise ValueError("Validation accuracy must be between 0 and 1.")
        runs.append({"path": str(root), "seed": seed, "records": records})

    curves = {}
    for name in METRICS:
        by_step = {}
        for run in runs:
            for record in run["records"]:
                if name in record:
                    by_step.setdefault(record["step"], []).append(record[name])
        curves[name] = [{"step": step, **summarize_values(values)} for step, values in sorted(by_step.items())]
    final = summarize_values([run["records"][-1]["val_accuracy"] for run in runs])
    summary = {
        "num_runs": len(runs), "seeds": [run["seed"] for run in runs],
        "run_paths": [run["path"] for run in runs],
        "config": reference, "final_val_accuracy": final,
        "four_seed_requirement_met": len(runs) == 4,
        "meets_25_percent_target": final["mean"] >= 0.25 if len(runs) == 4 else None,
        "interval_method": "Normal approximation: mean +/- 1.96 * sample_std / sqrt(n); not an exact small-sample interval.",
        "metrics": curves,
    }
    return runs, summary


def write_grpo_report(run_dirs: list[str | Path], output_dir: str | Path) -> Path:
    """Plot real logged metrics and create a report scaffold without inventing conclusions."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Report directory already exists: {output_dir}")
    runs, summary = aggregate_grpo_runs(run_dirs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            for run in runs:
                points = [record for record in run["records"] if name in record]
                ax.plot(
                    [point["step"] for point in points], [point[name] for point in points],
                    alpha=0.45, linewidth=1, label=f"seed {run['seed']}",
                )
            points = summary["metrics"][name]
            ax.plot([point["step"] for point in points], [point["mean"] for point in points],
                    color="black", linewidth=2, label="mean")
            band = [point for point in points if point["ci_lower"] is not None]
            if band:
                ax.fill_between([point["step"] for point in band], [point["ci_lower"] for point in band],
                                [point["ci_upper"] for point in band], color="gray", alpha=0.2,
                                label="normal-approx. 95% mean interval")
            ax.set(title=title, xlabel="Completed optimizer updates", ylabel=title)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"{name}.png", dpi=150)
        finally:
            plt.close(fig)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    lines = [
        "# Standard On-Policy GRPO Experiment", "",
        f"Model: `{summary['config']['model_id']}`. Seeds: {summary['seeds']}.",
        f"Final mean validation accuracy: {summary['final_val_accuracy']['mean']:.2%}.",
        f"Four-seed requirement met: {summary['four_seed_requirement_met']}.",
        f"25% target met: {summary['meets_25_percent_target']} (null means fewer/more than four seeds).",
        "", "Only completed, comparable runs are included. No conclusions are inferred from reward alone.",
        "Shading uses a normal approximation, not an exact confidence interval with four seeds.",
        "", "## Curves", "",
    ]
    for name, title in METRICS.items():
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
