import gzip
import json
from dataclasses import FrozenInstanceError

import pytest

from cs336_alignment.supplement.hh import HH_FILENAMES, HHPreferenceExample, load_hh_preferences


def conversation(instruction="Question", response="Answer"):
    return f"\n\nHuman: {instruction}\n\nAssistant: {response}"


def pair(**overrides):
    return {"chosen": conversation(response="Preferred"), "rejected": conversation(response="Other"), **overrides}


def write_files(root, examples):
    for filename in HH_FILENAMES:
        with gzip.open(root / filename, "wt", encoding="utf-8") as file:
            for example in examples.get(filename, []):
                file.write(json.dumps(example) + "\n")


def test_hh_combines_all_files_in_fixed_order_and_preserves_source(tmp_path):
    write_files(tmp_path, {
        filename: [pair(chosen=conversation(filename, "Preferred"), rejected=conversation(filename, "Other"))]
        for filename in HH_FILENAMES
    })
    actual = load_hh_preferences(tmp_path)
    assert actual == [HHPreferenceExample(filename, "Preferred", "Other", filename) for filename in HH_FILENAMES]
    assert load_hh_preferences(str(tmp_path)) == actual
    with pytest.raises(FrozenInstanceError):
        actual[0].chosen = "Changed"


@pytest.mark.parametrize("branches", [("chosen",), ("rejected",), ("chosen", "rejected")])
def test_hh_filters_pair_if_either_branch_has_multiple_human_messages(tmp_path, branches):
    multi_turn = pair(**{
        branch: conversation() + "\n\nHuman: Follow-up\n\nAssistant: Another answer" for branch in branches
    })
    write_files(tmp_path, {HH_FILENAMES[0]: [multi_turn, pair()]})
    assert load_hh_preferences(tmp_path) == [HHPreferenceExample("Question", "Preferred", "Other", HH_FILENAMES[0])]


def test_hh_keeps_response_after_first_assistant_marker_and_internal_formatting(tmp_path):
    chosen = "First paragraph.\n\nAssistant: Quoted continuation.\n  Indented line."
    rejected = "Human: is an ordinary word here, not a new turn."
    write_files(tmp_path, {HH_FILENAMES[0]: [pair(
        chosen=conversation("  Question  ", "  " + chosen + "  "),
        rejected=conversation(response=rejected),
    )]})
    assert load_hh_preferences(tmp_path) == [HHPreferenceExample("Question", chosen, rejected, HH_FILENAMES[0])]


@pytest.mark.parametrize("branch", ["chosen", "rejected"])
def test_hh_keeps_empty_responses_without_extra_filtering(tmp_path, branch):
    write_files(tmp_path, {HH_FILENAMES[0]: [pair(**{branch: conversation(response="")})]})
    examples = load_hh_preferences(tmp_path)
    assert len(examples) == 1
    assert getattr(examples[0], branch) == ""


@pytest.mark.parametrize("invalid, message", [
    ({"chosen": conversation()}, "chosen and rejected must be strings"),
    (pair(rejected=None), "chosen and rejected must be strings"),
    ([], "chosen and rejected must be strings"),
    (pair(chosen="Not a conversation"), "Human role marker"),
    (pair(rejected="\n\nHuman: No answer"), "Assistant role marker"),
    (pair(rejected=conversation("Different question")), "same instruction"),
])
def test_hh_reports_bad_single_turn_records_with_file_and_line(tmp_path, invalid, message):
    write_files(tmp_path, {HH_FILENAMES[0]: [pair(), invalid]})
    with pytest.raises(ValueError, match=message) as error:
        load_hh_preferences(tmp_path)
    assert f"{tmp_path / HH_FILENAMES[0]}: line 2:" in str(error.value)


def test_hh_reports_invalid_json_with_file_and_line(tmp_path):
    write_files(tmp_path, {})
    path = tmp_path / HH_FILENAMES[0]
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("{bad json}\n")
    with pytest.raises(ValueError) as error:
        load_hh_preferences(tmp_path)
    assert f"{path}: line 1:" in str(error.value)


def test_hh_does_not_silently_ignore_missing_files(tmp_path):
    write_files(tmp_path, {})
    (tmp_path / HH_FILENAMES[2]).unlink()
    with pytest.raises(FileNotFoundError):
        load_hh_preferences(tmp_path)


def test_hh_all_empty_files_return_empty_list(tmp_path):
    write_files(tmp_path, {})
    assert load_hh_preferences(tmp_path) == []
