import unittest

from cs336_alignment.prompting_evaluation import score_response


class TestPromptingEvaluation(unittest.TestCase):
    def test_correct_answers_receive_full_reward_for_all_prompt_styles(self):
        responses = {
            "question_only": r"The answer is \boxed{18}.",
            "r1_zero": "The result is 18.</think> <answer>18</answer>",
            "r1_zero_three_shot": "The result is 18.</think> <answer>18</answer>",
        }
        for prompt_name, response in responses.items():
            with self.subTest(prompt_name=prompt_name):
                self.assertEqual(
                    score_response(response, "18", prompt_name),
                    {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0},
                )

    def test_formatted_wrong_answers_do_not_receive_partial_credit(self):
        responses = {
            "question_only": r"\boxed{17}",
            "r1_zero": "</think> <answer>17</answer>",
            "r1_zero_three_shot": "</think> <answer>17</answer>",
        }
        for prompt_name, response in responses.items():
            with self.subTest(prompt_name=prompt_name):
                self.assertEqual(
                    score_response(response, "18", prompt_name),
                    {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0},
                )

    def test_correct_bare_number_is_unformatted_for_all_prompt_styles(self):
        for prompt_name in ("question_only", "r1_zero", "r1_zero_three_shot"):
            with self.subTest(prompt_name=prompt_name):
                self.assertEqual(
                    score_response("18", "18", prompt_name),
                    {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0},
                )

    def test_equivalent_numeric_answers_are_accepted(self):
        self.assertEqual(
            score_response(r"\boxed{1/2}", "0.5", "question_only")["answer_reward"],
            1.0,
        )

    def test_preserves_official_r1_format_requirement(self):
        self.assertEqual(
            score_response("</think><answer>18</answer>", "18", "r1_zero")["format_reward"],
            0.0,
        )

    def test_unknown_prompt_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown prompt name"):
            score_response("18", "18", "unknown")


if __name__ == "__main__":
    unittest.main()
