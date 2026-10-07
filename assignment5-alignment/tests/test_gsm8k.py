import json
import tempfile
import unittest
from pathlib import Path

from cs336_alignment.gsm8k import GSM8KExample, load_gsm8k


class TestGSM8K(unittest.TestCase):
    def test_extracts_final_answers_and_preserves_order(self):
        records = [
            {"question": " How many clips? ", "answer": "48 + 24 = 72\n#### 72 "},
            {"question": "Janet's income?", "answer": "Earlier #### text\n#### 18"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text("\n\n".join(json.dumps(record) for record in records), encoding="utf-8")
            self.assertEqual(
                load_gsm8k(path),
                [GSM8KExample("How many clips?", "72"), GSM8KExample("Janet's income?", "18")],
            )

    def test_rejects_records_that_cannot_be_scored(self):
        records = [
            [],
            {"question": "", "answer": "#### 72"},
            {"question": "Q", "answer": 72},
            {"question": "Q", "answer": "72"},
            {"question": "Q", "answer": "#### "},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            for record in records:
                with self.subTest(record=record):
                    path.write_text("\n" + json.dumps(record), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "invalid.jsonl:2:"):
                        load_gsm8k(path)

    def test_reads_bundled_test_set(self):
        path = Path(__file__).resolve().parents[1] / "data" / "gsm8k" / "test.jsonl"
        examples = load_gsm8k(path)
        self.assertEqual(len(examples), 1319)
        self.assertEqual(examples[0].ground_truth, "18")
        self.assertEqual(examples[1].ground_truth, "3")


if __name__ == "__main__":
    unittest.main()
