import pytest
import torch

from cs336_alignment.supplement.data import PackedSFTDataset, iterate_batches
from cs336_alignment.supplement.sft import compute_sft_loss, evaluate_sft, sft_train_step
from tests.common import FIXTURES_PATH


def make_batch(size):
    tokens = torch.arange(size * 4).reshape(size, 4) % 10 + 3
    return {"input_ids": tokens, "labels": (tokens + 1) % 15}


@pytest.mark.parametrize("training", [True, False])
def test_sft_validation_matches_full_batch_and_preserves_model(tiny_train_model, training):
    batches = [make_batch(2), make_batch(1)]
    full_batch = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
    tiny_train_model.eval()
    with torch.no_grad():
        expected_loss = compute_sft_loss(tiny_train_model, full_batch).item()
    tiny_train_model.train(training)
    parameters = [parameter.detach().clone() for parameter in tiny_train_model.parameters()]
    for parameter in tiny_train_model.parameters():
        parameter.grad = torch.ones_like(parameter)

    observed = []

    def inspect_forward(model, args):
        observed.append((model.training, torch.is_grad_enabled()))

    handle = tiny_train_model.register_forward_pre_hook(inspect_forward)
    try:
        metrics = evaluate_sft(tiny_train_model, iter(batches))
    finally:
        handle.remove()

    assert metrics["val_loss"] == pytest.approx(expected_loss)
    assert metrics["num_tokens"] == 12
    assert metrics["num_batches"] == 2
    assert observed == [(False, False), (False, False)]
    assert tiny_train_model.training == training
    assert torch.is_grad_enabled()
    for parameter, initial in zip(tiny_train_model.parameters(), parameters, strict=True):
        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)


def test_sft_validation_weights_by_token_count(tiny_train_model, monkeypatch):
    from cs336_alignment.supplement import sft

    losses = iter([torch.tensor(2.0), torch.tensor(8.0)])
    monkeypatch.setattr(sft, "compute_sft_loss", lambda model, batch: next(losses))
    metrics = evaluate_sft(tiny_train_model, [make_batch(2), make_batch(1)])
    # Eight tokens with loss 2, four with loss 8: token mean is 4, not batch mean 5.
    assert metrics == {"val_loss": 4.0, "num_tokens": 12, "num_batches": 2}


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("failure", ["empty", "invalid_shape", "source_error"])
def test_sft_validation_restores_mode_on_failure(tiny_train_model, training, failure):
    tiny_train_model.train(training)
    if failure == "empty":
        batches, error = [], ValueError
    elif failure == "invalid_shape":
        invalid = make_batch(1)
        invalid["labels"] = invalid["labels"][:, :-1]
        batches, error = [make_batch(2), invalid], ValueError
    else:
        def broken_source():
            yield make_batch(2)
            raise RuntimeError("validation data read failed")

        batches, error = broken_source(), RuntimeError
    with pytest.raises(error):
        evaluate_sft(tiny_train_model, batches)
    assert tiny_train_model.training == training
    assert torch.is_grad_enabled()
    assert all(parameter.grad is None for parameter in tiny_train_model.parameters())


def test_sft_training_can_continue_after_validation(tiny_train_model):
    optimizer = torch.optim.SGD(tiny_train_model.parameters(), lr=0.01)
    batches = [make_batch(2), make_batch(1)]
    sft_train_step(tiny_train_model, optimizer, batches)
    metrics = evaluate_sft(tiny_train_model, batches)
    assert tiny_train_model.training
    before = [parameter.detach().clone() for parameter in tiny_train_model.parameters()]
    loss, _ = sft_train_step(tiny_train_model, optimizer, batches)
    assert torch.isfinite(loss)
    assert metrics["num_tokens"] == 12
    assert any(not torch.equal(parameter, initial) for parameter, initial in zip(tiny_train_model.parameters(), before, strict=True))


def test_sft_validation_accepts_packed_dataloader(tiny_train_model, tokenizer):
    dataset = PackedSFTDataset(tokenizer, FIXTURES_PATH / "sft_sample.jsonl", seq_length=8, shuffle=False)
    loader = iterate_batches(dataset, batch_size=7, shuffle=False)
    metrics = evaluate_sft(tiny_train_model, loader)
    assert metrics["num_tokens"] == len(dataset) * 8
    assert metrics["num_batches"] == len(loader)
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
