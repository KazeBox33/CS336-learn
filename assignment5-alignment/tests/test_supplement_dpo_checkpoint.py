from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.supplement.checkpoint import save_best_dpo_checkpoint


@pytest.mark.parametrize("training", [True, False])
def test_dpo_checkpoint_keeps_earlier_ties_and_reloads_highest_accuracy(tiny_train_model, tokenizer, tmp_path, training):
    model = tiny_train_model
    model.train(training)
    output = tmp_path / "checkpoints"
    first_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best = save_best_dpo_checkpoint(model, tokenizer, accuracy=0.6, update=1, output_dir=output)
    first = best
    assert best.accuracy == 0.6 and best.update == 1
    assert best.path == output / "step_000001"
    assert model.training == training
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    for score, update in [(0.5, 2), (0.6, 3)]:
        best = save_best_dpo_checkpoint(model, tokenizer, accuracy=score, update=update, output_dir=output, best=best)
        assert best is first
        assert not (output / f"step_{update:06d}").exists()

    expected_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best = save_best_dpo_checkpoint(model, tokenizer, accuracy=0.8, update=4, output_dir=output, best=best)
    assert best.accuracy == 0.8 and best.update == 4
    assert best.path == output / "step_000004"
    assert set(output.iterdir()) == {first.path, best.path}
    assert model.training == training
    assert all(parameter.grad is None for parameter in model.parameters())
    reloaded = AutoModelForCausalLM.from_pretrained(best.path, local_files_only=True)
    earlier = AutoModelForCausalLM.from_pretrained(first.path, local_files_only=True)
    saved_tokenizer = AutoTokenizer.from_pretrained(best.path, local_files_only=True)
    assert saved_tokenizer.get_vocab() == tokenizer.get_vocab()
    for name, value in expected_state.items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(earlier.state_dict()[name], first_state[name], rtol=0, atol=0)
    assert not (best.path / "training_state.pt").exists()
    with pytest.raises(FrozenInstanceError):
        best.accuracy = 0.0


def test_first_dpo_validation_saves_even_zero_accuracy(tiny_train_model, tokenizer, tmp_path):
    best = save_best_dpo_checkpoint(tiny_train_model, tokenizer, accuracy=0.0, update=0, output_dir=str(tmp_path))
    assert best.accuracy == 0.0 and best.update == 0
    assert best.path == tmp_path / "step_000000"
    assert (best.path / "model.safetensors").is_file()


@pytest.mark.parametrize("accuracy, update", [(-0.1, 1), (1.1, 1), (float("nan"), 1), (float("inf"), 1), (0.5, -1), (0.5, 1.5)])
def test_dpo_checkpoint_rejects_invalid_metrics_before_writing(tmp_path, accuracy, update):
    model, tokenizer = Mock(), Mock()
    with pytest.raises(ValueError):
        save_best_dpo_checkpoint(model, tokenizer, accuracy=accuracy, update=update, output_dir=tmp_path / "checkpoints")
    model.save_pretrained.assert_not_called()
    tokenizer.save_pretrained.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["model", "tokenizer"])
def test_failed_dpo_export_preserves_previous_best(tiny_train_model, tokenizer, tmp_path, monkeypatch, failure):
    best = save_best_dpo_checkpoint(tiny_train_model, tokenizer, accuracy=0.5, update=1, output_dir=tmp_path)
    if failure == "model":
        monkeypatch.setattr(tiny_train_model, "save_pretrained", Mock(side_effect=RuntimeError("save failed")))
    else:
        monkeypatch.setattr(tokenizer, "save_pretrained", Mock(side_effect=RuntimeError("save failed")))
    with pytest.raises(RuntimeError, match="save failed"):
        best = save_best_dpo_checkpoint(tiny_train_model, tokenizer, accuracy=0.7, update=2, output_dir=tmp_path, best=best)
    assert best.accuracy == 0.5 and best.update == 1
    assert list(tmp_path.iterdir()) == [best.path]
    assert (best.path / "model.safetensors").is_file()


def test_dpo_checkpoint_refuses_to_overwrite_existing_step(tiny_train_model, tokenizer, tmp_path):
    best = save_best_dpo_checkpoint(tiny_train_model, tokenizer, accuracy=0.5, update=1, output_dir=tmp_path)
    with pytest.raises(FileExistsError):
        save_best_dpo_checkpoint(tiny_train_model, tokenizer, accuracy=0.7, update=1, output_dir=tmp_path, best=best)
    assert best.accuracy == 0.5
    assert list(tmp_path.iterdir()) == [best.path]
