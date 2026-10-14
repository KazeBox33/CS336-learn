import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from cs336_alignment import grpo_experiment
from cs336_alignment.grpo_experiment import GRPOExperimentData
from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.vllm_utils import VLLMCompletion


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
    spec = importlib.util.spec_from_file_location("train_grpo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_args(tmp_path):
    return [
        "--output-dir", str(tmp_path / "run"),
        "--n-train-examples", "2", "--n-val-examples", "1",
        "--num-rollout-steps", "2", "--rollout-batch-size", "4",
        "--group-size", "2", "--gradient-accumulation-steps", "2",
        "--eval-interval", "2", "--rollout-log-interval", "2", "--seed", "7",
    ]


def test_help_does_not_load_data_or_model(cli, monkeypatch, capsys):
    load = Mock()
    initialize = Mock()
    monkeypatch.setattr(cli, "load_experiment_data", load)
    monkeypatch.setattr(cli, "initialize_training_components", initialize)
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "--dry-run" in capsys.readouterr().out
    load.assert_not_called()
    initialize.assert_not_called()


def test_dry_run_loads_real_data_without_gpu_checks_or_output_files(cli, small_args, tmp_path, monkeypatch, capsys):
    initialize, open_server, available = Mock(), Mock(), Mock()
    monkeypatch.setattr(cli, "initialize_training_components", initialize)
    monkeypatch.setattr(cli, "open_rollout_server", open_server)
    monkeypatch.setattr(cli.torch.cuda, "is_available", available)
    cli.main([*small_args, "--dry-run"])
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] is True
    assert preview["completed"] is False
    assert preview["n_loaded_train_examples"] == 2
    assert preview["n_loaded_val_examples"] == 1
    assert preview["prompts_per_rollout_batch"] == 2
    assert preview["config"]["seed"] == 7
    assert Path(preview["config"]["train_path"]).is_absolute()
    initialize.assert_not_called()
    open_server.assert_not_called()
    available.assert_not_called()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("extra_args", [
    ["--rollout-batch-size", "3"],
    ["--validation-batch-size", "0"],
    ["--training-gpu", "1", "--inference-gpu", "1"],
    ["--inference-gpu", "-1"],
    ["--learning-rate", "nan"],
    ["--checkpoint-interval", "0"],
])
def test_invalid_arguments_fail_before_loading_data(cli, small_args, extra_args, monkeypatch):
    load = Mock()
    monkeypatch.setattr(cli, "load_experiment_data", load)
    with pytest.raises(SystemExit) as caught:
        cli.main([*small_args, *extra_args])
    assert caught.value.code == 2
    load.assert_not_called()


def test_existing_output_directory_is_preserved(cli, small_args, tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    existing = root / "metrics.jsonl"
    existing.write_text("previous experiment\n", encoding="utf-8")
    load = Mock()
    monkeypatch.setattr(cli, "load_experiment_data", load)
    with pytest.raises(SystemExit):
        cli.main(small_args)
    load.assert_not_called()
    assert existing.read_text(encoding="utf-8") == "previous experiment\n"


@pytest.mark.parametrize("available, count", [(False, 0), (True, 1)])
def test_missing_selected_gpus_prevents_model_loading(cli, small_args, tmp_path, monkeypatch, available, count):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: count)
    initialize = Mock()
    monkeypatch.setattr(cli, "initialize_training_components", initialize)
    with pytest.raises(SystemExit) as caught:
        cli.main(small_args)
    assert caught.value.code == 2
    initialize.assert_not_called()
    assert not (tmp_path / "run").exists()


def setup_fake_training(cli, monkeypatch):
    data = GRPOExperimentData(
        [GSM8KExample("Q", "1")], [GSM8KExample("V", "1")], ["TRAIN"], ["VAL"]
    )
    monkeypatch.setattr(cli, "load_experiment_data", Mock(return_value=data))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cli.torch.cuda, "device_count", lambda: 2)
    policy, tokenizer, optimizer, server = [Mock() for _ in range(4)]
    initialize = Mock(return_value=(policy, tokenizer, optimizer))
    monkeypatch.setattr(cli, "initialize_training_components", initialize)
    monkeypatch.setattr(cli, "save_grpo_checkpoint", Mock(side_effect=lambda *args, **kwargs: args[4]))
    events = []

    @contextmanager
    def open_server(config, model, *, inference_gpu):
        assert model is policy
        assert inference_gpu == 1
        events.append("start")
        try:
            yield server
        finally:
            events.append("stop")

    monkeypatch.setattr(cli, "open_rollout_server", open_server)
    return data, policy, tokenizer, optimizer, server, initialize, events


def test_cli_connects_loop_to_metrics_and_rollout_files(cli, small_args, tmp_path, monkeypatch, capsys):
    data, policy, tokenizer, optimizer, server, initialize, events = setup_fake_training(cli, monkeypatch)
    completion = VLLMCompletion("</think><answer>1</answer>", [1, 2], "stop")

    def train(config, passed_data, passed_server, model, passed_tokenizer, passed_optimizer, reward_fn, *, rng):
        assert passed_data is data and passed_server is server and model is policy
        assert passed_tokenizer is tokenizer and passed_optimizer is optimizer
        assert reward_fn is cli.r1_zero_reward_fn
        assert events == ["start"]
        return {"loss": torch.tensor(0.5)}, ["TRAIN"], [completion]

    train_mock = Mock(side_effect=train)
    validate_mock = Mock(return_value=({"val_accuracy": 1.0}, [completion]))
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", train_mock)
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", validate_mock)
    cli.main(small_args)
    root = tmp_path / "run"
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["completed"] is True
    assert manifest["num_metric_records"] == 3
    assert manifest["config"]["seed"] == 7
    assert manifest["training_device"] == "cuda:0"
    assert manifest["last_checkpoint"] == "checkpoints/step_0002"
    cli.save_grpo_checkpoint.assert_called_once()
    initialize.assert_called_once_with(initialize.call_args.args[0], "cuda:0")
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 1, 2]
    assert {path.name for path in (root / "rollouts").iterdir()} == {
        "val_step_0000.jsonl", "train_step_0001.jsonl", "train_step_0002.jsonl", "val_step_0002.jsonl"
    }
    assert train_mock.call_count == validate_mock.call_count == 2
    assert events == ["start", "stop"]
    assert json.loads(capsys.readouterr().out)["num_metric_records"] == 3


def test_training_failure_closes_server_and_does_not_mark_run_complete(cli, small_args, tmp_path, monkeypatch):
    *_, events = setup_fake_training(cli, monkeypatch)
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", Mock(return_value=(
        {"val_accuracy": 0.0}, [VLLMCompletion("wrong", [], "stop")]
    )))
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", Mock(side_effect=RuntimeError("training failed")))
    with pytest.raises(RuntimeError, match="training failed"):
        cli.main(small_args)
    root = tmp_path / "run"
    assert json.loads((root / "run.json").read_text())["completed"] is False
    assert json.loads((root / "metrics.jsonl").read_text())["step"] == 0
    assert events == ["start", "stop"]


def test_periodic_checkpoints_use_updated_policy_and_do_not_duplicate_final_step(cli, small_args, tmp_path, monkeypatch):
    _, policy, tokenizer, optimizer, _, _, events = setup_fake_training(cli, monkeypatch)
    completion = VLLMCompletion("answer", [], "stop")
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", Mock(return_value=({"loss": 0.5}, ["TRAIN"], [completion])))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", Mock(return_value=({"val_accuracy": 0.0}, [completion])))
    cli.main([*small_args, "--checkpoint-interval", "1"])
    calls = cli.save_grpo_checkpoint.call_args_list
    assert [item.args[3] for item in calls] == [1, 2]
    assert all(item.args[:3] == (policy, tokenizer, optimizer) for item in calls)
    assert calls[0].kwargs["rng"] is calls[1].kwargs["rng"]
    assert events == ["start", "stop"]


def test_checkpoint_failure_keeps_run_incomplete_and_closes_server(cli, small_args, tmp_path, monkeypatch):
    *_, events = setup_fake_training(cli, monkeypatch)
    completion = VLLMCompletion("answer", [], "stop")
    monkeypatch.setattr(grpo_experiment, "run_grpo_iteration", Mock(return_value=({"loss": 0.5}, ["TRAIN"], [completion])))
    monkeypatch.setattr(grpo_experiment, "evaluate_grpo_policy", Mock(return_value=({"val_accuracy": 0.0}, [completion])))
    cli.save_grpo_checkpoint.side_effect = RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        cli.main([*small_args, "--checkpoint-interval", "1"])
    assert json.loads((tmp_path / "run" / "run.json").read_text())["completed"] is False
    assert events == ["start", "stop"]
