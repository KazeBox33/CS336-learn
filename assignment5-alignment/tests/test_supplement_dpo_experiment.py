import gzip
import json
import math
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from cs336_alignment.supplement import dpo_experiment
from cs336_alignment.supplement.dpo_experiment import DPOTrainingConfig, run_dpo_training, split_hh_preferences
from cs336_alignment.supplement.hh import HH_FILENAMES, HHPreferenceExample, load_hh_preferences
from scripts.supplement import train_dpo


@pytest.fixture
def models(tokenizer):
    torch.manual_seed(17)
    policy = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=128, n_ctx=128, n_embd=8,
        n_layer=1, n_head=2, n_inner=16, use_cache=False,
    ))
    return policy, deepcopy(policy)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "hh"
    root.mkdir()
    for index, filename in enumerate(HH_FILENAMES):
        with gzip.open(root / filename, "wt", encoding="utf-8") as file:
            for offset in range(2):
                number = index * 2 + offset
                prompt = f"Question {number}"
                chosen, rejected = ("Hello", "Hello world") if number % 2 else ("Hello world", "Hello")
                file.write(json.dumps({
                    "chosen": f"\n\nHuman: {prompt}\n\nAssistant: {chosen}",
                    "rejected": f"\n\nHuman: {prompt}\n\nAssistant: {rejected}",
                }) + "\n")
    with gzip.open(root / HH_FILENAMES[0], "at", encoding="utf-8") as file:
        file.write(json.dumps({"chosen": "\n\nHuman: Question 0\n\nAssistant: Answer",
                               "rejected": "\n\nHuman: Question 0\n\nAssistant: Response"}) + "\n")
    return root


def config(**overrides):
    return DPOTrainingConfig(**{
        **dict(epochs=2, gradient_accumulation_steps=2, validation_size=2,
               learning_rate=1e-3, eval_interval=3), **overrides,
    })


def test_hh_split_is_reproducible_and_keeps_duplicate_instructions_together(data_dir):
    examples = load_hh_preferences(data_dir)
    random.seed(123)
    state = random.getstate()
    train, val = split_hh_preferences(examples, 2, 0)
    assert random.getstate() == state
    assert (train, val) == split_hh_preferences(examples, 2, 0)
    assert not {x.instruction for x in train} & {x.instruction for x in val}
    assert len({x.instruction for x in val}) == 2
    assert len(train) + len(val) == len(examples)
    for seed in range(10):
        train, val = split_hh_preferences(examples, 2, seed)
        assert sum(x.instruction == "Question 0" for x in train) in (0, 2)
        assert sum(x.instruction == "Question 0" for x in val) in (0, 2)


@pytest.mark.parametrize("size", [0, -1, 8, 9])
def test_hh_split_rejects_empty_training_or_validation(data_dir, size):
    with pytest.raises(ValueError, match="validation_size"):
        split_hh_preferences(load_hh_preferences(data_dir), size, 0)


@pytest.mark.parametrize("log_interval, eval_interval", [(1, 3), (100, 100)])
def test_dpo_experiment_connects_updates_logs_validation_and_best_checkpoint(
    models, tokenizer, data_dir, tmp_path, log_interval, eval_interval,
):
    policy, reference = models
    ref_state = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    cfg = config(log_interval=log_interval, eval_interval=eval_interval)
    output = tmp_path / "run"
    manifest = run_dpo_training(cfg, policy, reference, tokenizer, data_dir, output)
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    train_records = [x for x in records if x["split"] == "train"]
    val_records = [x for x in records if x["split"] == "validation"]
    total = cfg.epochs * math.ceil(manifest["num_train_examples"] / cfg.gradient_accumulation_steps)
    assert manifest == json.loads((output / "run.json").read_text())
    assert manifest["completed"] and manifest["optimizer_updates"] == manifest["total_updates"] == total
    assert manifest["num_train_examples"] + manifest["num_val_examples"] == 9
    assert manifest["num_val_instructions"] == 2
    assert [x["update"] for x in train_records] == [i for i in range(1, total + 1) if i % log_interval == 0 or i == total]
    assert [x["update"] for x in val_records] == [0] + [i for i in range(1, total + 1) if i % eval_interval == 0 or i == total]
    if log_interval == 1:
        assert sum(x["num_examples"] for x in train_records) == manifest["num_train_examples"] * cfg.epochs
    assert manifest["final_val_accuracy"] == val_records[-1]["val_accuracy"]
    assert manifest["best_accuracy"] == max(x["val_accuracy"] for x in val_records)
    assert manifest["best_update"] == next(x["update"] for x in val_records if x["val_accuracy"] == manifest["best_accuracy"])
    assert (Path(manifest["best_model_dir"]) / "model.safetensors").is_file()
    assert AutoTokenizer.from_pretrained(manifest["best_model_dir"], local_files_only=True).get_vocab() == tokenizer.get_vocab()
    for name, value in ref_state.items():
        torch.testing.assert_close(reference.state_dict()[name], value, rtol=0, atol=0)
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())


def test_dpo_best_checkpoint_can_precede_final_update(models, tokenizer, data_dir, tmp_path, monkeypatch):
    policy, reference = models
    accuracies = iter([0.4, 0.8, 0.2])
    monkeypatch.setattr(dpo_experiment, "evaluate_dpo", lambda *args: {
        "val_loss": 1.0, "val_accuracy": next(accuracies), "num_examples": 2,
    })
    snapshots = {}
    original = dpo_experiment.save_best_dpo_checkpoint

    def save(model, tok, **kwargs):
        snapshots[kwargs["update"]] = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        return original(model, tok, **kwargs)

    monkeypatch.setattr(dpo_experiment, "save_best_dpo_checkpoint", save)
    manifest = run_dpo_training(config(epochs=1, gradient_accumulation_steps=4, eval_interval=1),
                                policy, reference, tokenizer, data_dir, tmp_path / "run")
    assert manifest["optimizer_updates"] == 2
    assert manifest["best_update"] == 1 and manifest["best_accuracy"] == 0.8
    saved = AutoModelForCausalLM.from_pretrained(manifest["best_model_dir"], local_files_only=True)
    for name, value in snapshots[1].items():
        torch.testing.assert_close(saved.state_dict()[name], value, rtol=0, atol=0)
    assert any(not torch.equal(snapshots[1][name], snapshots[2][name]) for name in snapshots[1])


def test_dpo_experiment_is_reproducible(models, tokenizer, data_dir, tmp_path):
    policy, reference = models
    runs = []
    for number in range(2):
        output = tmp_path / str(number)
        run_dpo_training(config(epochs=1), deepcopy(policy), deepcopy(reference), tokenizer, data_dir, output)
        runs.append((output / "metrics.jsonl").read_text())
    assert runs[0] == runs[1]


@pytest.mark.parametrize("settings", [
    {"epochs": 0}, {"gradient_accumulation_steps": 0}, {"validation_size": 0},
    {"log_interval": 0}, {"eval_interval": 0}, {"seed": -1}, {"learning_rate": 0},
    {"beta": -1}, {"max_grad_norm": -1}, {"learning_rate": float("nan")},
])
def test_dpo_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        DPOTrainingConfig(**settings)


def test_dpo_experiment_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        run_dpo_training(config(), Mock(), Mock(), Mock(), "missing", tmp_path)


def test_dpo_experiment_rejects_shared_models_before_freezing(models, tokenizer, data_dir, tmp_path):
    policy, _ = models
    with pytest.raises(ValueError, match="Parameter"):
        run_dpo_training(config(), policy, policy, tokenizer, data_dir, tmp_path / "run")
    assert all(p.requires_grad for p in policy.parameters())


def test_dpo_training_failure_keeps_best_record_and_incomplete_manifest(models, tokenizer, data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(dpo_experiment, "run_dpo_epoch", Mock(side_effect=RuntimeError("update failed")))
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="update failed"):
        run_dpo_training(config(), *models, tokenizer, data_dir, output)
    manifest = json.loads((output / "run.json").read_text())
    assert not manifest["completed"] and manifest["best_update"] == 0
    assert (Path(manifest["best_model_dir"]) / "model.safetensors").is_file()


def test_dpo_cli_trains_two_local_cpu_models(models, tokenizer, data_dir, tmp_path):
    source = tmp_path / "sft_model"
    models[0].save_pretrained(source)
    tokenizer.save_pretrained(source)
    output = tmp_path / "cli_run"
    result = subprocess.run([
        sys.executable, "-m", "scripts.supplement.train_dpo", "--model-id", str(source),
        "--data-dir", str(data_dir), "--output-dir", str(output), "--local-files-only",
        "--validation-size", "2", "--gradient-accumulation-steps", "2", "--eval-interval", "2",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["completed"]
    assert manifest["policy_device"] == manifest["reference_device"] == "cpu"
    assert json.loads(result.stdout.splitlines()[-1])["best_model_dir"] == manifest["best_model_dir"]


def test_dpo_cli_rejects_bad_settings_and_missing_data_before_loading(tmp_path, monkeypatch):
    load = Mock()
    monkeypatch.setattr(train_dpo.AutoTokenizer, "from_pretrained", load)
    base = ["--model-id", "unused", "--data-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "run")]
    for extra in (["--epochs", "0"], []):
        with pytest.raises(SystemExit) as error:
            train_dpo.main(base + extra)
        assert error.value.code == 2
    load.assert_not_called()
