from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.supplement.checkpoint import save_sft_model
from cs336_alignment.supplement.sft import sft_train_step


@pytest.mark.parametrize("training", [True, False])
def test_sft_export_round_trips_trained_model_and_tokenizer(tiny_train_model, tokenizer, tmp_path, training):
    optimizer = torch.optim.AdamW(tiny_train_model.parameters(), lr=0.01)
    batch = {"input_ids": torch.tensor([[3, 4, 5, 6]]), "labels": torch.tensor([[4, 5, 6, 7]])}
    sft_train_step(tiny_train_model, optimizer, [batch])
    tiny_train_model.train(training)
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    initial = {name: tensor.detach().clone() for name, tensor in tiny_train_model.state_dict().items()}

    destination = tmp_path / "nested" / "sft_final"
    result = save_sft_model(tiny_train_model, tokenizer, destination)
    assert result == destination
    assert (result / "config.json").is_file()
    assert (result / "model.safetensors").is_file()
    assert (result / "tokenizer_config.json").is_file()
    assert not (result / "training_state.pt").exists()
    assert list(destination.parent.iterdir()) == [destination]
    assert tiny_train_model.training == training
    assert all(parameter.grad is None for parameter in tiny_train_model.parameters())

    reloaded = AutoModelForCausalLM.from_pretrained(result, local_files_only=True)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(result, local_files_only=True)
    assert reloaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert reloaded_tokenizer.eos_token_id == tokenizer.eos_token_id
    assert reloaded_tokenizer.pad_token_id == tokenizer.pad_token_id
    assert reloaded_tokenizer.chat_template == tokenizer.chat_template
    assert reloaded_tokenizer.encode("Hello world") == tokenizer.encode("Hello world")
    for name, expected in initial.items():
        torch.testing.assert_close(tiny_train_model.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(reloaded.state_dict()[name], expected, rtol=0, atol=0)
    tiny_train_model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            tiny_train_model(input_ids=batch["input_ids"]).logits,
            reloaded(input_ids=batch["input_ids"]).logits,
            rtol=0, atol=0,
        )


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_sft_export_refuses_existing_destination(tmp_path, kind):
    destination = tmp_path / "existing"
    if kind == "directory":
        destination.mkdir()
        (destination / "keep.txt").write_text("keep", encoding="utf-8")
    elif kind == "file":
        destination.write_text("keep", encoding="utf-8")
    else:
        destination.symlink_to(tmp_path / "missing")
    model, tokenizer = Mock(), Mock()
    with pytest.raises(FileExistsError, match="already exists"):
        save_sft_model(model, tokenizer, destination)
    model.save_pretrained.assert_not_called()
    tokenizer.save_pretrained.assert_not_called()
    if kind == "dangling_symlink":
        assert destination.is_symlink()
    else:
        saved = destination / "keep.txt" if kind == "directory" else destination
        assert saved.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("failure", ["model", "tokenizer"])
def test_sft_export_cleans_partial_output_on_failure(tmp_path, failure):
    destination = tmp_path / "sft_final"
    model, tokenizer = Mock(), Mock()

    def save_partial(path, **kwargs):
        (Path(path) / "partial.bin").write_bytes(b"partial")
        if failure == "model":
            raise RuntimeError("model save failed")

    model.save_pretrained.side_effect = save_partial
    tokenizer.save_pretrained.side_effect = RuntimeError("tokenizer save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        save_sft_model(model, tokenizer, destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
    if failure == "model":
        tokenizer.save_pretrained.assert_not_called()


def test_sft_export_accepts_string_path(tiny_train_model, tokenizer, tmp_path):
    destination = tmp_path / "sft_final"
    assert save_sft_model(tiny_train_model, tokenizer, str(destination)) == destination
