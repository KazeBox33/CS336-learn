import random
from unittest.mock import Mock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.checkpoint import save_grpo_checkpoint


def test_checkpoint_round_trip_preserves_weights_tokenizer_optimizer_and_rng(tmp_path, tiny_train_model, tokenizer):
    model = tiny_train_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(input_ids=torch.tensor([[1, 2]])).logits.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(7)
    rng.random()
    path = save_grpo_checkpoint(model, tokenizer, optimizer, 1, tmp_path / "step_0001", rng=rng)
    restored = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    restored_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    assert restored_tokenizer.encode("Hello world") == tokenizer.encode("Hello world")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    assert state["step"] == 1
    resumed_optimizer = torch.optim.AdamW(restored.parameters())
    resumed_optimizer.load_state_dict(state["optimizer"])
    for index, original in optimizer.state_dict()["state"].items():
        for name, value in original.items():
            torch.testing.assert_close(resumed_optimizer.state_dict()["state"][index][name], value)
    resumed_rng = random.Random()
    resumed_rng.setstate(state["rollout_rng"])
    assert resumed_rng.random() == rng.random()


def test_existing_checkpoint_is_not_modified(tmp_path):
    root = tmp_path / "step_0001"
    root.mkdir()
    marker = root / "original"
    marker.write_text("keep", encoding="utf-8")
    model = Mock()
    with pytest.raises(FileExistsError):
        save_grpo_checkpoint(model, Mock(), Mock(), 1, root)
    assert marker.read_text() == "keep"
    model.save_pretrained.assert_not_called()


def test_failed_checkpoint_is_not_published_and_temporary_directory_is_cleaned(tmp_path):
    model = Mock()
    model.save_pretrained.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        save_grpo_checkpoint(model, Mock(), Mock(), 1, tmp_path / "step_0001")
    assert list(tmp_path.iterdir()) == []


def test_negative_checkpoint_step_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        save_grpo_checkpoint(Mock(), Mock(), Mock(), -1, tmp_path / "bad")
    assert list(tmp_path.iterdir()) == []
