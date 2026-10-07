import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from cs336_alignment.gsm8k import GSM8KExample
from cs336_alignment.prompting import PROMPT_FILES
from cs336_alignment.prompting_baselines import evaluate_prompt, run_prompting_baselines
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer
from scripts.evaluate_prompting import main


def make_server():
    server = Mock(spec=VLLMServer)
    server.model_id = "TEST_MODEL"
    return server


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestPromptingBaselines(unittest.TestCase):
    def test_runs_all_prompts_and_saves_metrics_responses_and_review(self):
        server = make_server()
        examples = [GSM8KExample(f"QUESTION_{index}", "18") for index in range(3)]
        server.generate_completions.side_effect = [
            [
                VLLMCompletion(r"\boxed{18}", [10], "stop"),
                VLLMCompletion(r"\boxed{17}", [20], "stop"),
                VLLMCompletion("18", [30], "length"),
            ],
            *[
                [
                    VLLMCompletion("</think> <answer>18</answer>", [10], "stop"),
                    VLLMCompletion("</think> <answer>17</answer>", [20], "stop"),
                    VLLMCompletion("18", [30], "length"),
                ]
                for _ in range(2)
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = run_prompting_baselines(
                server, examples, root, seed=42, batch_size=2
            )
            self.assertEqual(json.loads((root / "summary.json").read_text()), summary)
            self.assertTrue(summary["completed"])
            self.assertEqual(list(summary["prompts"]), list(PROMPT_FILES))
            for name, result in summary["prompts"].items():
                metrics = result["metrics"]
                self.assertEqual(metrics["num_responses"], 3)
                self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
                self.assertAlmostEqual(metrics["format_rate"], 2 / 3)
                self.assertEqual(metrics["num_length_limited"], 1)
                self.assertEqual(result["review_counts"], {"1": 1, "2": 1, "3": 1})
                records = read_jsonl(root / result["responses_file"])
                self.assertEqual([record["category"] for record in records], [1, 2, 3])
                self.assertEqual(
                    [record["example_index"] for record in records], [0, 1, 2]
                )
                self.assertEqual(records[0]["token_ids"], [10])
                self.assertIn("QUESTION_0", records[0]["prompt"])
                self.assertEqual(result["sampling_params"]["seed"], 42)
                review = read_jsonl(root / result["review_file"])
                self.assertTrue(
                    all(record["human_answer_correct"] is None for record in review)
                )
                self.assertTrue(
                    all(
                        record["parser_missed_correct_answer"] is None
                        for record in review
                    )
                )
                self.assertEqual(review[0]["response"], records[0]["response"])
            report = (root / "report.md").read_text()
            self.assertIn("Manual Analysis (Pending)", report)
            self.assertIn("33.33%", report)
        self.assertEqual(server.generate_completions.call_count, 3)
        for call in server.generate_completions.call_args_list:
            self.assertEqual(call.kwargs["batch_size"], 2)
        server.start.assert_not_called()
        server.stop.assert_not_called()

    def test_selects_ten_examples_per_failure_category_when_available(self):
        server = make_server()
        responses = [r"\boxed{18}"] * 3 + [r"\boxed{17}"] * 12 + ["18"] * 10
        server.generate_completions.return_value = [
            VLLMCompletion(text, [], "stop") for text in responses
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = run_prompting_baselines(
                server,
                [GSM8KExample("QUESTION", "18")] * 25,
                directory,
                prompt_names=["question_only"],
            )
            result = summary["prompts"]["question_only"]
            self.assertEqual(result["review_counts"], {"1": 3, "2": 10, "3": 10})
            review = read_jsonl(Path(directory) / result["review_file"])
            self.assertEqual(len(review), 23)
            self.assertEqual(
                [row["example_index"] for row in review if row["category"] == 2],
                list(range(3, 13)),
            )

    def test_existing_results_are_not_overwritten(self):
        server = make_server()
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "summary.json"
            saved.write_text("KEEP_THIS_RESULT")
            with self.assertRaises(FileExistsError):
                run_prompting_baselines(
                    server, [GSM8KExample("QUESTION", "18")], directory
                )
            self.assertEqual(saved.read_text(), "KEEP_THIS_RESULT")
        server.generate_completions.assert_not_called()

    def test_failure_preserves_completed_prompt_results(self):
        server = make_server()
        server.generate_completions.side_effect = [
            [VLLMCompletion(r"\boxed{18}", [], "stop")],
            RuntimeError("FAILED"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "FAILED"):
                run_prompting_baselines(
                    server, [GSM8KExample("QUESTION", "18")], directory
                )
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertFalse(summary["completed"])
            self.assertEqual(list(summary["prompts"]), ["question_only"])
            self.assertTrue(
                (Path(directory) / "question_only.responses.jsonl").exists()
            )
            self.assertIn(
                "Completed: False", (Path(directory) / "report.md").read_text()
            )

    def test_invalid_inputs_do_not_generate_responses(self):
        server = make_server()
        examples = [GSM8KExample("QUESTION", "18")]
        with tempfile.TemporaryDirectory() as directory:
            for names in ([], ["unknown"], ["r1_zero", "r1_zero"]):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    run_prompting_baselines(
                        server, examples, directory, prompt_names=names
                    )
            with self.assertRaises(ValueError):
                evaluate_prompt(server, [], "question_only")
            with self.assertRaises(ValueError):
                evaluate_prompt(server, examples, "question_only", batch_size=0)
        server.generate_completions.assert_not_called()


class TestPromptingCLI(unittest.TestCase):
    def test_full_cli_with_local_http_endpoint_and_real_grader(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def reply(self, body):
                encoded = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                self.reply({})

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                requests.append(payload)
                choices = []
                for index, prompt in enumerate(payload["prompt"]):
                    question_index = int(
                        re.search(r"TEST_QUESTION_(\d+)", prompt).group(1)
                    )
                    answer = "18" if question_index == 0 else "17"
                    text = (
                        rf"\boxed{{{answer}}}"
                        if "stop" not in payload
                        else f"</think> <answer>{answer}</answer>"
                    )
                    if question_index == 2:
                        text = "18"
                    choices.append(
                        {
                            "index": index,
                            "text": text,
                            "token_ids": [question_index],
                            "finish_reason": "length"
                            if question_index == 2
                            else "stop",
                        }
                    )
                self.reply({"choices": list(reversed(choices))})

            def log_message(self, *args):
                pass

        endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / "test.jsonl"
                data.write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "question": f"TEST_QUESTION_{index}",
                                "answer": "Calculation #### 18",
                            }
                        )
                        for index in range(3)
                    )
                )
                with patch("builtins.print"):
                    main(
                        [
                            "--model-id",
                            "TEST_MODEL",
                            "--data-path",
                            str(data),
                            "--output-dir",
                            str(root / "output"),
                            "--connect-only",
                            "--port",
                            str(endpoint.server_port),
                            "--batch-size",
                            "2",
                        ]
                    )
                summary = json.loads((root / "output/summary.json").read_text())
                self.assertTrue(summary["completed"])
                for name, result in summary["prompts"].items():
                    self.assertAlmostEqual(result["metrics"]["accuracy"], 1 / 3)
                    records = read_jsonl(root / "output" / result["responses_file"])
                    self.assertEqual(
                        [record["category"] for record in records], [1, 2, 3]
                    )
                    self.assertEqual(
                        [record["token_ids"] for record in records], [[0], [1], [2]]
                    )
            self.assertEqual(len(requests), 6)
            self.assertNotIn("stop", requests[0])
            self.assertNotIn("stop", requests[1])
            for payload in requests[2:]:
                self.assertEqual(payload["stop"], ["</answer>"])
                self.assertTrue(payload["include_stop_str_in_output"])
        finally:
            endpoint.shutdown()
            endpoint.server_close()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_connect_only_runs_selected_prompt_and_leaves_server_running(self):
        server = make_server()
        server.generate_completions.return_value = [
            VLLMCompletion(r"\boxed{18}", [], "stop")
        ] * 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.jsonl"
            data_path.write_text(
                "\n".join(
                    json.dumps(
                        {"question": f"Q{index}", "answer": "Calculation #### 18"}
                    )
                    for index in range(3)
                )
            )
            output = root / "results"
            with (
                patch(
                    "cs336_alignment.vllm_utils.VLLMServer", return_value=server
                ) as constructor,
                patch("builtins.print"),
            ):
                main(
                    [
                        "--model-id",
                        "TEST_MODEL",
                        "--data-path",
                        str(data_path),
                        "--output-dir",
                        str(output),
                        "--prompts",
                        "question_only",
                        "--limit",
                        "2",
                        "--seed",
                        "7",
                        "--connect-only",
                    ]
                )
            constructor.assert_called_once()
            self.assertFalse(constructor.call_args.kwargs["launch_server"])
            self.assertEqual(constructor.call_args.kwargs["seed"], 7)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["num_examples"], 2)
            self.assertEqual(summary["data_path"], str(data_path.resolve()))
            self.assertEqual(summary["requested_prompts"], ["question_only"])
            self.assertEqual(
                summary["prompts"]["question_only"]["metrics"]["accuracy"], 1.0
            )
        server.start.assert_called_once()
        server.stop.assert_not_called()

    def test_launched_server_is_stopped_even_if_generation_fails(self):
        server = make_server()
        server.generate_completions.side_effect = RuntimeError("FAILED")
        with tempfile.TemporaryDirectory() as directory:
            with patch("cs336_alignment.vllm_utils.VLLMServer", return_value=server):
                with self.assertRaisesRegex(RuntimeError, "FAILED"):
                    main(
                        [
                            "--model-id",
                            "TEST_MODEL",
                            "--output-dir",
                            directory,
                            "--limit",
                            "1",
                        ]
                    )
        server.start.assert_called_once()
        server.stop.assert_called_once()

    def test_launched_server_is_stopped_if_startup_fails(self):
        server = make_server()
        server.start.side_effect = RuntimeError("STARTUP_FAILED")
        with tempfile.TemporaryDirectory() as directory:
            with patch("cs336_alignment.vllm_utils.VLLMServer", return_value=server):
                with self.assertRaisesRegex(RuntimeError, "STARTUP_FAILED"):
                    main(
                        [
                            "--model-id",
                            "TEST_MODEL",
                            "--output-dir",
                            directory,
                            "--limit",
                            "1",
                        ]
                    )
        server.stop.assert_called_once()
        server.generate_completions.assert_not_called()

    def test_invalid_limit_is_rejected_before_server_creation(self):
        with (
            patch("cs336_alignment.vllm_utils.VLLMServer") as constructor,
            patch("sys.stderr"),
        ):
            with self.assertRaises(SystemExit) as error:
                main(
                    [
                        "--model-id",
                        "TEST_MODEL",
                        "--output-dir",
                        "unused",
                        "--limit",
                        "0",
                    ]
                )
        self.assertEqual(error.exception.code, 2)
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
