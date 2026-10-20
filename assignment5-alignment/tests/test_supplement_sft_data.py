import json

import pytest

from cs336_alignment.supplement.data import load_sft_documents

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
            f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}\n"
        )


def test_load_sft_documents_reports_invalid_fields_without_skipping_samples(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"prompt": "Question", "response": "Answer"}\n{"prompt": "Missing answer"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: prompt and response must be strings") as error:
        load_sft_documents(path)
    assert str(path) in str(error.value)
