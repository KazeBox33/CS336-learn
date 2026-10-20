import json
from collections import Counter

import pytest
import torch
from transformers import AutoTokenizer

from cs336_alignment.supplement.data import (
    PackedSFTDataset,
    iterate_batches,
    load_sft_documents,
    tokenize_sft_documents,
)

from .common import FIXTURES_PATH


def test_load_sft_documents_uses_official_template_and_preserves_content():
    path = FIXTURES_PATH / "sft_sample.jsonl"
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    documents = load_sft_documents(path)
    assert len(documents) == len(examples)
    for document, example in zip(documents, examples, strict=True):
        assert document == (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
        )


def test_load_sft_documents_reports_invalid_fields_without_skipping_samples(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"prompt": "Question", "response": "Answer"}\n{"prompt": "Missing answer"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: prompt and response must be strings") as error:
        load_sft_documents(path)
    assert str(path) in str(error.value)


def test_tokenize_sft_documents_matches_official_tokens_and_document_boundaries():
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B", local_files_only=True)
    documents = load_sft_documents(FIXTURES_PATH / "sft_sample.jsonl")
    token_ids = tokenize_sft_documents(documents, tokenizer)
    expected_examples = json.loads((FIXTURES_PATH / "tokenized_sft_sample.json").read_text(encoding="utf-8"))
    seq_length = len(expected_examples[0]["input_ids"])
    for i, example in enumerate(expected_examples):
        start = i * seq_length
        assert token_ids[start : start + seq_length] == example["input_ids"]
        assert token_ids[start + 1 : start + seq_length + 1] == example["labels"]
    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert token_ids.count(tokenizer.bos_token_id) == len(documents)
    assert token_ids.count(tokenizer.eos_token_id) == len(documents)


def test_packed_sft_dataset_reserves_next_token_and_checks_bounds():
    path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B", local_files_only=True)
    token_ids = tokenize_sft_documents(load_sft_documents(path), tokenizer)
    dataset = PackedSFTDataset(tokenizer, path, seq_length=len(token_ids) - 1, shuffle=False)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"].tolist() == token_ids[:-1]
    assert dataset[0]["labels"].tolist() == token_ids[1:]
    assert dataset[0]["input_ids"].dtype == torch.long
    assert dataset[0]["labels"].dtype == torch.long
    with pytest.raises(IndexError):
        dataset[1]
    with pytest.raises(IndexError):
        dataset[-1]
    empty = PackedSFTDataset(tokenizer, path, seq_length=len(token_ids), shuffle=False)
    assert len(empty) == 0
    with pytest.raises(IndexError):
        empty[0]
    with pytest.raises(ValueError, match="seq_length must be positive"):
        PackedSFTDataset(tokenizer, path, seq_length=0, shuffle=False)


def test_iterate_batches_preserves_pairs_and_visits_every_item_once():
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B", local_files_only=True)
    dataset = PackedSFTDataset(tokenizer, FIXTURES_PATH / "sft_sample.jsonl", seq_length=32, shuffle=False)
    expected = Counter((tuple(item["input_ids"].tolist()), tuple(item["labels"].tolist())) for item in dataset)
    for shuffle in (False, True):
        loader = iterate_batches(dataset, batch_size=8, shuffle=shuffle)
        batches = list(loader)
        assert len(batches) == len(loader) == 10
        assert batches[-1]["input_ids"].shape == (3, 32)
        assert batches[-1]["labels"].shape == (3, 32)
        actual = Counter(
            (tuple(input_ids.tolist()), tuple(labels.tolist()))
            for batch in batches
            for input_ids, labels in zip(batch["input_ids"], batch["labels"], strict=True)
        )
        assert actual == expected
        if not shuffle:
            assert torch.equal(
                torch.cat([batch["input_ids"] for batch in batches]),
                torch.stack([item["input_ids"] for item in dataset]),
            )
