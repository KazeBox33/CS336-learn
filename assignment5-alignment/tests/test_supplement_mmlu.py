import pytest

from cs336_alignment.supplement.evaluation import parse_mmlu_response


@pytest.mark.parametrize(
    "model_output, expected",
    [
        ("The correct answer is D", "D"),
        ("the correct answer is b. Here is the explanation.", "B"),
        ("The correct answer is C\n```", "C"),
        ("The correct answer is a national holiday.", None),
        ("The correct answer is Biology.", None),
        ("The correct answer is E.", None),
        ("A and B both seem possible.", None),
    ],
)
def test_parse_mmlu_response_respects_answer_format(model_output, expected):
    assert parse_mmlu_response(model_output) == expected
