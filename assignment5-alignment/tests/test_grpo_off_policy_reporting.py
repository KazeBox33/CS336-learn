import json
from unittest.mock import Mock

import pytest

from cs336_alignment import grpo_reporting
from cs336_alignment.grpo_reporting import aggregate_grpo_runs, write_grpo_comparison, write_grpo_report
from scripts import run_grpo_off_policy, summarize_grpo
from tests.test_grpo_reporting import make_run


def make_logged_run(root, seed=0, method="grpo", on_policy=False):
    make_run(root, seed=seed, accuracy=0.2 + 0.05 * seed)
    path = root / "run.json"
    manifest = json.loads(path.read_text())
    config = manifest["config"]
    config.update(
        rollout_batch_size=256, train_batch_size=256 if on_policy else 8,
        gradient_accumulation_steps=32 if on_policy else 1,
        importance_reweighting_method=method,
        cliprange=0.2 if method == "grpo" else 3e-4 if method == "gspo" else None,
        learning_rate=1e-5,
    )
    path.write_text(json.dumps(manifest))
    path = root / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        step = record["step"]
        record.update(elapsed_seconds=2.0 + step * 5.0 + seed, optimizer_updates=step * (1 if on_policy else 32))
        if step:
            record["train_iteration_seconds"] = 4.0
            if method in ("grpo", "gspo"):
                record["train_clip_fraction"] = 0.1 * step
        if "val_accuracy" in record:
            record["val_seconds"] = 1.0
    path.write_text("\n".join(json.dumps(record) for record in records))
    return root


def test_off_policy_summary_tracks_clip_units_and_both_step_counters(tmp_path):
    roots = [make_logged_run(tmp_path / f"seed_{seed}", seed=seed, method="gspo") for seed in range(4)]
    _, summary = aggregate_grpo_runs(roots)
    assert summary["optimizer_steps_per_rollout"] == 32
    assert summary["clip_fraction_unit"] == "response"
    points = summary["metrics"]["train_clip_fraction"]
    assert [point["optimizer_updates"] for point in points] == [32, 64]
    assert [point["elapsed_seconds"] for point in points] == [8.5, 13.5]
    assert all(point["n"] == 4 for point in points)


@pytest.mark.parametrize("kind", ["missing_clip", "bad_clip", "negative_time", "decreasing_time", "missing_time", "wrong_updates"])
def test_invalid_off_policy_logs_are_rejected(tmp_path, kind):
    root = make_logged_run(tmp_path / "run")
    path = root / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if kind == "missing_clip":
        records[1].pop("train_clip_fraction")
    elif kind == "bad_clip":
        records[1]["train_clip_fraction"] = 1.1
    elif kind == "negative_time":
        records[1]["train_iteration_seconds"] = -1.0
    elif kind == "decreasing_time":
        records[-1]["elapsed_seconds"] = 0.0
    elif kind == "missing_time":
        records[1].pop("elapsed_seconds")
    else:
        records[1]["optimizer_updates"] = 1
    path.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError):
        aggregate_grpo_runs([root])


@pytest.mark.parametrize("axis", ["rollout_batches", "optimizer_updates", "elapsed_seconds"])
def test_comparison_groups_seeds_without_mixing_algorithms_and_writes_real_plots(tmp_path, axis):
    pytest.importorskip("matplotlib")
    from PIL import Image
    roots = []
    for name, method, on_policy in [
        ("onpolicy", "none", True), ("naive", "none", False),
        ("noclip", "noclip", False), ("clip", "grpo", False), ("gspo", "gspo", False),
    ]:
        for seed in range(4):
            roots.append(make_logged_run(tmp_path / f"{name}_{seed}", seed, method, on_policy))
    report = write_grpo_comparison(roots, tmp_path / "report", x_axis=axis)
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["x_axis"] == axis
    assert summary["four_method_four_seed_requirement_met"] is True
    assert summary["onpolicy_baseline_included"] is True
    assert set(summary["methods"]) == {"onpolicy", "offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo"}
    assert all(stats["four_seed_requirement_met"] for stats in summary["methods"].values())
    assert summary["methods"]["onpolicy"]["optimizer_steps_per_rollout"] == 1
    assert "Analysis pending" in report.read_text()
    assert "not directly comparable" in report.read_text()
    for name in ("train_clip_fraction", "train_iteration_seconds", "val_accuracy"):
        with Image.open(report.parent / f"{name}.png") as image:
            assert image.width >= 900
            assert any(low < high for low, high in image.convert("RGB").getextrema())


def test_comparison_rejects_changed_learning_rate_before_creating_report(tmp_path):
    first = make_logged_run(tmp_path / "clip")
    second = make_logged_run(tmp_path / "gspo", method="gspo")
    path = second / "run.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["learning_rate"] = 0.01
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hyperparameters"):
        write_grpo_comparison([first, second], tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_comparison_rejects_changed_runtime_settings(tmp_path):
    first = make_logged_run(tmp_path / "clip")
    second = make_logged_run(tmp_path / "gspo", method="gspo")
    path = second / "run.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_interval"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint"):
        write_grpo_comparison([first, second], tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_legacy_logs_support_batch_and_update_axes_but_not_elapsed_time(tmp_path, monkeypatch):
    root = make_run(tmp_path / "legacy")
    plot = Mock()
    monkeypatch.setattr(grpo_reporting, "_plot_metric", plot)
    write_grpo_report([root], tmp_path / "batch_report")
    assert all(call.args[-1] == "rollout_batches" for call in plot.call_args_list)
    with pytest.raises(ValueError, match="timing"):
        write_grpo_report([root], tmp_path / "time_report", x_axis="elapsed_seconds")
    assert not (tmp_path / "time_report").exists()


def test_summarize_cli_selects_comparison_and_axis(monkeypatch, tmp_path):
    writer = Mock(return_value=tmp_path / "report.md")
    monkeypatch.setattr(summarize_grpo, "write_grpo_comparison", writer)
    summarize_grpo.main(["--run-dirs", str(tmp_path / "one"), str(tmp_path / "two"),
                         "--output-dir", str(tmp_path / "report"), "--compare-methods", "--x-axis", "elapsed_seconds"])
    assert writer.call_args.kwargs == {"x_axis": "elapsed_seconds"}


def test_experiment_plan_defaults_to_16_commands_without_launching_or_writing(tmp_path, monkeypatch, capsys):
    run = Mock()
    monkeypatch.setattr(run_grpo_off_policy.subprocess, "run", run)
    root = tmp_path / "experiments"
    run_grpo_off_policy.main(["--output-root", str(root), "--", "--learning-rate", "2e-5"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 16
    assert all("--train-batch-size 8" in line and "--gradient-accumulation-steps 1" in line for line in lines)
    assert all("--learning-rate 2e-5" in line for line in lines)
    assert sum("--cliprange 0.2" in line for line in lines) == 4
    assert sum("--cliprange 3e-4" in line for line in lines) == 4
    run.assert_not_called()
    assert not root.exists()


def test_experiment_plan_optionally_includes_four_on_policy_runs(tmp_path):
    commands = run_grpo_off_policy.build_commands(tmp_path, [0, 1, 2, 3], [], True)
    assert len(commands) == 20
    for command in commands[:4]:
        assert command[command.index("--train-batch-size") + 1] == "256"
        assert command[command.index("--gradient-accumulation-steps") + 1] == "32"


def test_experiment_plan_rejects_duplicate_seeds_and_existing_outputs(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        run_grpo_off_policy.build_commands(tmp_path, [0, 0], [])
    (tmp_path / "offpolicy_gspo" / "seed_0").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_grpo_off_policy.build_commands(tmp_path, [0], [])


@pytest.mark.parametrize("extra", [["--seed", "9"], ["--cliprange=0.1"], ["--dry-run"]])
def test_experiment_plan_rejects_overridden_controlled_options(tmp_path, extra):
    with pytest.raises(ValueError, match="controlled"):
        run_grpo_off_policy.build_commands(tmp_path, [0], extra)


def test_experiment_execution_is_explicit_sequential_and_stops_on_failure(tmp_path, monkeypatch):
    import subprocess
    run = Mock(side_effect=[None, subprocess.CalledProcessError(1, "training")])
    monkeypatch.setattr(run_grpo_off_policy.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        run_grpo_off_policy.main(["--output-root", str(tmp_path / "runs"), "--seeds", "0", "--execute"])
    assert run.call_count == 2
    assert all(call.kwargs["check"] is True for call in run.call_args_list)
