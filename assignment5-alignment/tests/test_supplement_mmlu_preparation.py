import csv
from pathlib import Path

import pytest

from cs336_alignment.supplement.mmlu import MMLUExample, load_mmlu
from cs336_alignment.supplement.prompting import build_mmlu_prompts


def write_subject(root, subject, rows, split="test"):
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{subject}_{split}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)
    return path


def test_loader_preserves_quoted_commas_newlines_and_stable_subject_order(tmp_path):
    write_subject(tmp_path, "zoology", [["Last question", "1", "2", "3", "4", "D"]])
    write_subject(tmp_path, "computer_security", [
        ["Which choice, and why?\nSecond line.", "First, option", "B", "C", "D", "A"],
        ["Next question", "a", "b", "c", "d", "B"],
    ])
    examples = load_mmlu(tmp_path)
    assert examples == [
        MMLUExample("computer_security", "Which choice, and why?\nSecond line.", ("First, option", "B", "C", "D"), "A"),
        MMLUExample("computer_security", "Next question", ("a", "b", "c", "d"), "B"),
        MMLUExample("zoology", "Last question", ("1", "2", "3", "4"), "D"),
    ]


def test_loader_defaults_to_test_and_keeps_dev_examples_separate(tmp_path):
    write_subject(tmp_path, "history", [["Test question", "a", "b", "c", "d", "A"]])
    write_subject(tmp_path, "history", [["Dev question", "a", "b", "c", "d", "B"]], split="dev")
    assert [example.question for example in load_mmlu(tmp_path)] == ["Test question"]
    assert [example.question for example in load_mmlu(tmp_path, split="dev")] == ["Dev question"]


@pytest.mark.parametrize("row", [
    ["Q", "a", "b", "c", "A"],
    ["Q", "a", "b", "c", "d", "E"],
    ["Q", "a", "b", "c", "d", ""],
    ["   ", "a", "b", "c", "d", "A"],
    ["Q", "a", " ", "c", "d", "A"],
])
def test_loader_reports_invalid_row_with_file_and_row_number(tmp_path, row):
    path = write_subject(tmp_path, "history", [[], row])
    with pytest.raises(ValueError, match="row 2") as error:
        load_mmlu(tmp_path)
    assert str(path) in str(error.value)


def test_loader_rejects_empty_split_missing_files_and_unknown_split(tmp_path):
    with pytest.raises(FileNotFoundError, match="No MMLU test"):
        load_mmlu(tmp_path)
    write_subject(tmp_path, "history", [])
    with pytest.raises(ValueError, match="No MMLU examples"):
        load_mmlu(tmp_path)
    with pytest.raises(ValueError, match="split"):
        load_mmlu(tmp_path, split="train")


def test_loader_reads_the_bundled_test_set():
    root = Path(__file__).resolve().parents[1] / "data" / "mmlu"
    examples = load_mmlu(root)
    assert len(examples) == 14042
    assert len({example.subject for example in examples}) == 57
    assert examples[0].subject == "abstract_algebra"
    assert examples[0].options == ("0", "4", "2", "6")
    assert examples[0].answer == "B"


def test_prompts_use_both_official_templates_without_reference_answer_leakage():
    examples = [
        MMLUExample("computer_security", "What is {1, 2}?", ("one", "two", "three", "four"), answer)
        for answer in ("A", "D")
    ]
    prompts = build_mmlu_prompts(iter(examples))
    assert len(prompts) == 2
    assert prompts[0] == prompts[1]
    prompt = prompts[0]
    assert prompt.startswith("# Instruction\n")
    assert "question about computer security." in prompt
    assert "Question: What is {1, 2}?\nA. one\nB. two\nC. three\nD. four\nAnswer:\n" in prompt
    assert "\n# Query:\n```Answer the following" in prompt
    assert prompt.endswith("\n# Answer:\n```\n")
    assert prompt.count("\nQuestion:") == 1


def test_prompts_preserve_example_order_and_allow_empty_input():
    examples = [MMLUExample("history", question, ("a", "b", "c", "d"), "A") for question in ("FIRST", "SECOND")]
    prompts = build_mmlu_prompts(examples)
    assert "Question: FIRST\n" in prompts[0]
    assert "Question: SECOND\n" in prompts[1]
    assert build_mmlu_prompts([]) == []
