from pathlib import Path
from collections import Counter
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def train_bpe(
        input_path:str|Path,
        vocab_size:int,
        special_tokens:list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab = {}

    for i in range(256):
        vocab[i]=bytes([i])

    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")
    
    merges=[]

    text=Path(input_path).read_text(encoding="utf-8")
    special_pattern="|".join(re.escape(token) for token in special_tokens)

    if special_tokens:
        pass
    
    return vocab,merges
