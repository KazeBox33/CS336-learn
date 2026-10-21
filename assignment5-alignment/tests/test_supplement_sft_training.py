import gzip
import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.supplement import sft_training
from cs336_alignment.supplement.data import PackedSFTDataset, iterate_batches, load_sft_documents
from cs336_alignment.supplement.sft import evaluate_sft
from cs336_alignment.supplement.sft_training import SFTTrainingConfig, run_sft_training
from scripts.supplement import train_sft


@pytest.fixture
def data_paths(tmp_path):
    paths = []
    for split, examples in (
        ("train", [{"prompt": "Hello world", "response": "This is a test"}] * 3),
        ("val", [{"prompt": "Question", "response": "Answer"}]),
    ):
        path = tmp_path / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as file:
            for example in examples:
                file.write(json.dumps(example) + "\n")
        paths.append(path)
    return paths


def small_config(**overrides):
    settings = dict(epochs=2, seq_length=8, microbatch_size=3, gradient_accumulation_steps=2,
                    learning_rate=0.01, log_interval=1, eval_interval=2)
    return SFTTrainingConfig(**{**settings, **overrides})


@pytest.mark.parametrize("log_interval, eval_interval", [(1, 2), (100, 100)])
def test_sft_training_connects_updates_validation_logs_and_export(
    tiny_train_model, tokenizer, data_paths, tmp_path, log_interval, eval_interval,
):
    initial = [parameter.detach().clone() for parameter in tiny_train_model.parameters()]
    config = small_config(log_interval=log_interval, eval_interval=eval_interval)
    output = tmp_path / "run"
    manifest = run_sft_training(config, tiny_train_model, tokenizer, *data_paths, output)
    saved_manifest = json.loads((output / "run.json").read_text())
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "validation"]
    num_microbatches = math.ceil(manifest["num_train_sequences"] / config.microbatch_size)
    updates_per_epoch = math.ceil(num_microbatches / config.gradient_accumulation_steps)
    total = config.epochs * updates_per_epoch
    assert manifest == saved_manifest
    assert manifest["completed"]
    assert manifest["optimizer_updates"] == manifest["total_updates"] == total
    assert manifest["updates_per_epoch"] == updates_per_epoch
    assert manifest["effective_batch_size"] == 6
    assert manifest["warmup_updates"] == min(math.ceil(total * config.warmup_ratio), total - 1)
    assert [record["update"] for record in train_records] == [
        update for update in range(1, total + 1) if update % log_interval == 0 or update == total
    ]
    assert [record["update"] for record in val_records] == [0] + [
        update for update in range(1, total + 1) if update % eval_interval == 0 or update == total
    ]
    assert all(record["num_tokens"] == manifest["num_val_sequences"] * 8 for record in val_records)
    if log_interval == 1:
        assert sum(record["num_tokens"] for record in train_records) == manifest["num_train_sequences"] * 8 * config.epochs
        assert sum(record["num_microbatches"] for record in train_records) == num_microbatches * config.epochs
        assert [record["epoch"] for record in train_records] == [1] * updates_per_epoch + [2] * updates_per_epoch
        assert train_records[0]["learning_rate"] == 0.0
        assert train_records[updates_per_epoch]["learning_rate"] > 0.0
    assert any(not torch.equal(parameter, before) for parameter, before in zip(tiny_train_model.parameters(), initial, strict=True))

    model = AutoModelForCausalLM.from_pretrained(output / "final", local_files_only=True)
    saved_tokenizer = AutoTokenizer.from_pretrained(output / "final", local_files_only=True)
    dataset = PackedSFTDataset(saved_tokenizer, data_paths[1], seq_length=8, shuffle=False)
    metrics = evaluate_sft(model, iterate_batches(dataset, 3, shuffle=False))
    assert manifest["final_val_loss"] == pytest.approx(metrics["val_loss"])
    assert manifest["final_val_loss"] == val_records[-1]["val_loss"]
    assert Path(manifest["model_dir"]) == output / "final"


def test_sft_training_repeats_with_same_seed(tiny_train_model, tokenizer, data_paths, tmp_path):
    models = [deepcopy(tiny_train_model), deepcopy(tiny_train_model)]
    metrics = []
    for index, model in enumerate(models):
        output = tmp_path / f"run_{index}"
        run_sft_training(small_config(epochs=1), model, tokenizer, *data_paths, output)
        metrics.append((output / "metrics.jsonl").read_text())
    assert metrics[0] == metrics[1]
    for first, second in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("settings", [
    {"epochs": 0}, {"seq_length": 0}, {"microbatch_size": -1}, {"gradient_accumulation_steps": 0},
    {"log_interval": 0}, {"eval_interval": 0}, {"seed": -1}, {"learning_rate": 0},
    {"weight_decay": -1}, {"max_grad_norm": -1}, {"warmup_ratio": 1}, {"warmup_ratio": -0.1},
    {"learning_rate": float("nan")}, {"max_grad_norm": float("inf")},
])
def test_sft_training_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        SFTTrainingConfig(**settings)


def test_sft_training_refuses_existing_output_before_reading_data(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_sft_training(small_config(), Mock(), Mock(), "missing_train", "missing_val", output)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("split", [0, 1])
def test_sft_training_rejects_empty_packed_dataset(tiny_train_model, tokenizer, data_paths, tmp_path, split):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    data_paths[split] = empty
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="at least one packed sequence"):
        run_sft_training(small_config(), tiny_train_model, tokenizer, *data_paths, output)
    assert not output.exists()


def test_sft_training_failure_is_not_marked_complete(tiny_train_model, tokenizer, data_paths, tmp_path, monkeypatch):
    monkeypatch.setattr(sft_training, "run_sft_epoch", Mock(side_effect=RuntimeError("training failed")))
    save = Mock()
    monkeypatch.setattr(sft_training, "save_sft_model", save)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="training failed"):
        run_sft_training(small_config(), tiny_train_model, tokenizer, *data_paths, output)
    assert not json.loads((output / "run.json").read_text())["completed"]
    assert not (output / "final").exists()
    save.assert_not_called()


def test_sft_plain_and_gzip_jsonl_match(data_paths, tmp_path):
    plain = tmp_path / "train.jsonl"
    with gzip.open(data_paths[0], "rt", encoding="utf-8") as file:
        plain.write_text(file.read(), encoding="utf-8")
    assert load_sft_documents(plain) == load_sft_documents(data_paths[0])


@pytest.mark.parametrize("option, value", [("--epochs", "0"), ("--warmup-ratio", "1")])
def test_sft_cli_rejects_bad_settings_before_loading(option, value, tmp_path, monkeypatch):
    load_model, load_tokenizer = Mock(), Mock()
    monkeypatch.setattr(train_sft.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(train_sft.AutoTokenizer, "from_pretrained", load_tokenizer)
    with pytest.raises(SystemExit) as error:
        train_sft.main(["--model-id", "unused", "--train-path", "unused", "--val-path", "unused",
                        "--output-dir", str(tmp_path / "run"), option, value])
    assert error.value.code == 2
    load_model.assert_not_called()
    load_tokenizer.assert_not_called()


def test_sft_cli_trains_local_cpu_model(tiny_train_model, tokenizer, data_paths, tmp_path):
    source = tmp_path / "source_model"
    tiny_train_model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    output = tmp_path / "cli_run"
    result = subprocess.run([
        sys.executable, "-m", "scripts.supplement.train_sft", "--model-id", str(source),
        "--train-path", str(data_paths[0]), "--val-path", str(data_paths[1]), "--output-dir", str(output),
        "--local-files-only", "--device", "cpu", "--seq-length", "8", "--microbatch-size", "3",
        "--gradient-accumulation-steps", "2", "--learning-rate", "0.01", "--log-interval", "1",
    ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["completed"]
    assert manifest["device"] == "cpu"
    assert (output / "final" / "model.safetensors").is_file()
    assert json.loads(result.stdout.splitlines()[-1])["final_val_loss"] == manifest["final_val_loss"]


def test_sft_single_update_disables_warmup(tiny_train_model, tokenizer, data_paths, tmp_path):
    manifest = run_sft_training(
        small_config(epochs=1, gradient_accumulation_steps=1000),
        tiny_train_model, tokenizer, *data_paths, tmp_path / "one_update",
    )
    assert manifest["optimizer_updates"] == 1
    assert manifest["warmup_updates"] == 0


def test_sft_cli_rejects_missing_data_before_loading(tmp_path, monkeypatch):
    load_tokenizer = Mock()
    monkeypatch.setattr(train_sft.AutoTokenizer, "from_pretrained", load_tokenizer)
    with pytest.raises(SystemExit) as error:
        train_sft.main(["--model-id", "unused", "--train-path", str(tmp_path / "missing_train"),
                        "--val-path", str(tmp_path / "missing_val"), "--output-dir", str(tmp_path / "run")])
    assert error.value.code == 2
    load_tokenizer.assert_not_called()
