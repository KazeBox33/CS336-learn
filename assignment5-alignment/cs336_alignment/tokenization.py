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
