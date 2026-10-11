import torch
from transformers import PreTrainedTokenizerBase


def _tokenize_pairs(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[list[int]], list[int]]:
    """Concatenate separately tokenized pairs and retain each prompt's length."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("Each prompt must have exactly one output.")
    if not prompt_strs:
        raise ValueError("The batch must contain at least one prompt-output pair.")

    prompt_ids = tokenizer(
        prompt_strs, add_special_tokens=False, padding=False, truncation=False
    )["input_ids"]
    output_ids = tokenizer(
        output_strs, add_special_tokens=False, padding=False, truncation=False
    )["input_ids"]

    token_sequences = []
    prompt_lengths = []
    for prompt_tokens, output_tokens in zip(prompt_ids, output_ids, strict=True):
        token_sequences.append(prompt_tokens + output_tokens)
        prompt_lengths.append(len(prompt_tokens))
    return token_sequences, prompt_lengths


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Right-pad token pairs and align response masks with next-token labels."""
    token_sequences, prompt_lengths = _tokenize_pairs(prompt_strs, output_strs, tokenizer)
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad_token_id.")

    max_length = max(len(tokens) for tokens in token_sequences)
    if max_length == 0:
        raise ValueError("The batch must contain at least one token.")

    batch_shape = (len(token_sequences), max_length)
    token_ids = torch.full(batch_shape, tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros(batch_shape, dtype=torch.bool)

    for row, (tokens, prompt_length) in enumerate(
        zip(token_sequences, prompt_lengths, strict=True)
    ):
        sequence_length = len(tokens)
        token_ids[row, :sequence_length] = torch.tensor(tokens, dtype=torch.long)
        response_mask[row, prompt_length:sequence_length] = True # mask 标记回答为true

    # Shift the mask with labels so the first response prediction is retained.
    return {
        "input_ids": token_ids[:, :-1],
        "labels": token_ids[:, 1:],
        "response_mask": response_mask[:, 1:],
    }
