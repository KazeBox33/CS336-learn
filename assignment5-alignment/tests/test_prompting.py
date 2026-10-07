import unittest

from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.prompting import PROMPT_FILES, build_prompts


class TestPrompting(unittest.TestCase):
    def test_question_only_has_literal_latex_and_no_reference_answer(self):
        example = GSM8KExample("Find x in {1, 2}.", "REFERENCE_ANSWER")
        self.assertEqual(
            build_prompts([example], "question_only"),
            [r"Find x in {1, 2}. Please put your final answer within \boxed{}."],
        )

    def test_r1_templates_keep_demonstrations_and_open_thinking_prefix(self):
        for prompt_name, demonstrations in [("r1_zero", 0), ("r1_zero_three_shot", 3)]:
            with self.subTest(prompt_name=prompt_name):
                prompt = build_prompts([GSM8KExample("CURRENT_QUESTION", "REFERENCE_ANSWER")], prompt_name)[0]
                self.assertEqual(prompt.count("\nUser:"), demonstrations + 1)
                self.assertTrue(prompt.endswith("User: CURRENT_QUESTION\nAssistant: <think>"))
                self.assertNotIn("REFERENCE_ANSWER", prompt)

    def test_preserves_question_order_and_accepts_an_iterator(self):
        examples = [GSM8KExample("FIRST_QUESTION", "1"), GSM8KExample("SECOND_QUESTION", "2")]
        for prompt_name in PROMPT_FILES:
            with self.subTest(prompt_name=prompt_name):
                prompts = build_prompts(iter(examples), prompt_name)
                self.assertEqual(len(prompts), 2)
                self.assertIn("FIRST_QUESTION", prompts[0])
                self.assertIn("SECOND_QUESTION", prompts[1])

    def test_rejects_unknown_prompt_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown prompt name"):
            build_prompts([], "unknown")


if __name__ == "__main__":
    unittest.main()
