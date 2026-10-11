import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from cs336_alignment.tokenization import _tokenize_pairs


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


if __name__ == "__main__":
    unittest.main()
