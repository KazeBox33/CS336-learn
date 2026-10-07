import unittest
from unittest.mock import Mock, patch

from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.prompting_generation import generate_responses
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


class TestPromptingGeneration(unittest.TestCase):
    def test_connects_templates_and_sampling_to_existing_server(self):
        server = VLLMServer(model_id="TEST_MODEL", launch_server=False)
        examples = [GSM8KExample("FIRST_QUESTION", "18"), GSM8KExample("SECOND_QUESTION", "3")]
        replies = [
            {"choices": [{"index": 0, "text": "</think> <answer>18</answer>", "token_ids": [10], "finish_reason": "stop"}]},
            {"choices": [{"index": 0, "text": "</think> <answer>3</answer>", "token_ids": [20], "finish_reason": "stop"}]},
        ]
        with patch("cs336_alignment.vllm_utils._http_json", side_effect=replies) as request:
            completions = generate_responses(server, examples, "r1_zero", seed=42, batch_size=1)
        self.assertEqual([reply.text for reply in completions], [item["choices"][0]["text"] for item in replies])
        self.assertEqual(request.call_count, 2)
        for call, example in zip(request.call_args_list, examples):
            payload = call.args[2]
            self.assertEqual(payload["model"], "TEST_MODEL")
            self.assertEqual(len(payload["prompt"]), 1)
            self.assertTrue(payload["prompt"][0].endswith(f"User: {example.question}\nAssistant: <think>"))
            self.assertEqual(payload["n"], 1)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["stop"], ["</answer>"])
            self.assertTrue(payload["include_stop_str_in_output"])

    def test_rejects_response_count_mismatch(self):
        server = Mock(spec=VLLMServer)
        server.generate_completions.return_value = []
        with self.assertRaisesRegex(RuntimeError, "Expected 1 responses, received 0"):
            generate_responses(server, [GSM8KExample("QUESTION", "18")], "question_only")

    def test_returns_server_results_without_losing_token_ids(self):
        server = Mock(spec=VLLMServer)
        results = [VLLMCompletion(r"\boxed{18}", [10, 20], "stop")]
        server.generate_completions.return_value = results
        self.assertIs(generate_responses(server, [GSM8KExample("QUESTION", "18")], "question_only"), results)

    def test_empty_data_does_not_send_a_generation_request(self):
        server = Mock(spec=VLLMServer)
        self.assertEqual(generate_responses(server, [], "r1_zero"), [])
        server.generate_completions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
