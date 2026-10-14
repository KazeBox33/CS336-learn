import json
import math

import pytest

from cs336_alignment.grpo_reporting import (
    METRICS, TRAIN_METRICS, VAL_METRICS, aggregate_grpo_runs, write_grpo_report,
)


def make_run(root, seed=0, accuracy=0.25, completed=True):
    root.mkdir()
    config = {
        "model_id": "test/model", "seed": seed, "num_rollout_steps": 2,
        "eval_interval": 2, "train_path": "/data/train.jsonl", "val_path": "/data/test.jsonl",
    }
    (root / "run.json").write_text(json.dumps({"config": config, "completed": completed}), encoding="utf-8")
    records = [
        {"step": 0, **{name: 0.125 for name in VAL_METRICS}},
        {"step": 1, **{name: 0.5 for name in TRAIN_METRICS}},
        {"step": 2, **{name: 0.25 for name in METRICS}},
    ]
    records[-1]["val_accuracy"] = accuracy
    (root / "metrics.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return root


def test_four_seed_statistics_and_exact_step_alignment(tmp_path):
    roots = [make_run(tmp_path / f"seed_{seed}", seed, accuracy) for seed, accuracy in enumerate([0.2, 0.3, 0.4, 0.5])]
    runs, summary = aggregate_grpo_runs(roots)
    assert len(runs) == 4
    assert summary["four_seed_requirement_met"] is True
    assert summary["meets_25_percent_target"] is True
    final = summary["final_val_accuracy"]
    assert final["mean"] == pytest.approx(0.35)
    assert final["sample_std"] == pytest.approx(math.sqrt(0.05 / 3))
    assert final["ci_upper"] - final["mean"] == pytest.approx(1.96 * final["sample_std"] / 2)
    assert [point["step"] for point in summary["metrics"]["train_loss"]] == [1, 2]
    assert [point["step"] for point in summary["metrics"]["val_accuracy"]] == [0, 2]
    assert all(point["n"] == 4 for point in summary["metrics"]["val_accuracy"])


def test_single_run_does_not_claim_four_seed_target_or_zero_uncertainty(tmp_path):
    _, summary = aggregate_grpo_runs([make_run(tmp_path / "one", accuracy=0.9)])
    assert summary["meets_25_percent_target"] is None
    assert summary["four_seed_requirement_met"] is False
    assert summary["final_val_accuracy"]["sample_std"] is None
    assert summary["final_val_accuracy"]["ci_lower"] is None


def test_target_uses_mean_final_accuracy_not_the_best_seed(tmp_path):
    roots = [make_run(tmp_path / f"seed_{seed}", seed, value) for seed, value in enumerate([0.1, 0.2, 0.2, 0.4])]
    _, summary = aggregate_grpo_runs(roots)
    assert summary["final_val_accuracy"]["mean"] == pytest.approx(0.225)
    assert summary["meets_25_percent_target"] is False


@pytest.mark.parametrize("kind", ["incomplete", "duplicate_seed", "different_config"])
def test_incomplete_or_incomparable_runs_are_rejected(tmp_path, kind):
    first = make_run(tmp_path / "one", seed=0)
    second = make_run(tmp_path / "two", seed=0 if kind == "duplicate_seed" else 1, completed=kind != "incomplete")
    if kind == "different_config":
        manifest_path = second / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["model_id"] = "other/model"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        aggregate_grpo_runs([first, second])


@pytest.mark.parametrize("kind", ["missing_step", "duplicate_step", "missing_metric", "nan", "invalid_accuracy"])
def test_malformed_metric_logs_are_rejected(tmp_path, kind):
    root = make_run(tmp_path / "one")
    path = root / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if kind == "missing_step":
        records.pop(1)
    elif kind == "duplicate_step":
        records[1]["step"] = 0
    elif kind == "missing_metric":
        del records[1]["train_grad_norm"]
    elif kind == "nan":
        records[1]["train_loss"] = float("nan")
    else:
        records[-1]["val_accuracy"] = 1.1
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError):
        aggregate_grpo_runs([root])


def test_report_writes_nonblank_plots_summary_and_analysis_placeholders(tmp_path):
    pytest.importorskip("matplotlib")
    from PIL import Image
    roots = [make_run(tmp_path / f"seed_{seed}", seed, 0.2 + 0.1 * seed) for seed in range(4)]
    report = write_grpo_report(roots, tmp_path / "report")
    summary = json.loads((report.parent / "summary.json").read_text())
    assert summary["num_runs"] == 4
    assert "Analysis pending" in report.read_text()
    assert "before and after training" in report.read_text()
    assert "35.00%" in report.read_text()
    for name in METRICS:
        path = report.parent / f"{name}.png"
        assert path.stat().st_size > 1000
        with Image.open(path) as image:
            assert image.width >= 900
            extrema = image.convert("RGB").getextrema()
            assert any(low < high for low, high in extrema)
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        write_grpo_report(roots, report.parent)
    assert report.read_bytes() == before
