import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from cs336_alignment.tokenization import _tokenize_pairs, tokenize_prompt_and_output


class TestTokenizePairs(unittest.TestCase):
    def setUp(self):
        backend = Tokenizer(
            WordLevel(
                {"<pad>": 0, "<bos>": 1, "<unk>": 2, "Hello": 3, "world": 4, "!": 5},
                unk_token="<unk>",
            )
        )
        backend.pre_tokenizer = Whitespace()
        backend.post_processor = TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", 1)]
        )
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", unk_token="<unk>"
        )

    def test_preserves_variable_lengths_without_padding_or_truncation(self):
        sequences, prompt_lengths = _tokenize_pairs(
            ["Hello", "Hello world"], ["world !", ""], self.tokenizer
        )
        self.assertEqual(sequences, [[3, 4, 5], [3, 4]])
        self.assertEqual(prompt_lengths, [1, 2])

    def test_tokenizes_before_concatenation(self):
        sequences, _ = _tokenize_pairs(["Hello"], ["world"], self.tokenizer)
        self.assertEqual(sequences, [[3, 4]])
        self.assertNotEqual(
            sequences[0], self.tokenizer.encode("Helloworld", add_special_tokens=False)
        )

    def test_disables_automatic_special_tokens(self):
        self.assertEqual(self.tokenizer.encode("Hello"), [1, 3])
        sequences, _ = _tokenize_pairs(["Hello"], ["world"], self.tokenizer)
        self.assertEqual(sequences, [[3, 4]])

    def test_rejects_unpaired_or_empty_batches(self):
        for prompts, outputs in [(["Hello"], []), ([], [])]:
            with self.subTest(prompts=prompts, outputs=outputs):
                with self.assertRaises(ValueError):
                    _tokenize_pairs(prompts, outputs, self.tokenizer)

    def test_pads_and_shifts_tokens_and_response_mask(self):
        result = tokenize_prompt_and_output(
            ["Hello world", "Hello"], ["world !", "!"], self.tokenizer
        )
        self.assertEqual(result["input_ids"].tolist(), [[3, 4, 4], [3, 5, 0]])
        self.assertEqual(result["labels"].tolist(), [[4, 4, 5], [5, 0, 0]])
        self.assertEqual(
            result["response_mask"].tolist(), [[False, True, True], [True, False, False]]
        )
        self.assertEqual(result["input_ids"].dtype, torch.long)
        self.assertEqual(result["labels"].dtype, torch.long)
        self.assertEqual(result["response_mask"].dtype, torch.bool)

    def test_empty_response_has_no_response_positions(self):
        result = tokenize_prompt_and_output(["Hello world"], [""], self.tokenizer)
        self.assertEqual(result["labels"].tolist(), [[4]])
        self.assertEqual(result["response_mask"].tolist(), [[False]])

    def test_response_mask_uses_positions_not_token_values(self):
        result = tokenize_prompt_and_output(["Hello"], ["<pad> !"], self.tokenizer)
        self.assertEqual(result["labels"].tolist(), [[0, 5]])
        self.assertEqual(result["response_mask"].tolist(), [[True, True]])

    def test_right_padding_does_not_depend_on_tokenizer_padding_side(self):
        self.tokenizer.padding_side = "left"
        result = tokenize_prompt_and_output(
            ["Hello world", "Hello"], ["world !", "!"], self.tokenizer
        )
        self.assertEqual(result["labels"].tolist(), [[4, 4, 5], [5, 0, 0]])

    def test_single_token_sequence_has_no_next_token_pairs(self):
        result = tokenize_prompt_and_output(["Hello"], [""], self.tokenizer)
        for tensor in result.values():
            self.assertEqual(tensor.shape, (1, 0))

    def test_rejects_missing_padding_token_or_all_empty_sequences(self):
        with self.assertRaisesRegex(ValueError, "at least one token"):
            tokenize_prompt_and_output([""], [""], self.tokenizer)
        self.tokenizer.pad_token = None
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            tokenize_prompt_and_output(["Hello"], ["world"], self.tokenizer)


if __name__ == "__main__":
    unittest.main()
