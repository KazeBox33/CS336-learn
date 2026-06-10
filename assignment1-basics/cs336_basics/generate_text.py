from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def load_pickle(path:str|Path):
    with open(path,"rb") as file:
        return pickle.load(file)

def main() -> None:
    parser =argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--merges-path", required=True)
    parser.add_argument("--prompt", required=True)

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default="cpu")

    args=parser.parse_args()

    vocab=load_pickle(args.vocab_path)
    merges=load_pickle(args.merges_path)
    tokenizer=Tokenizer(vocab,merges,special_tokens=["<|endoftext|>"])

    device=torch.device(args.device)

    model=TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    checkpoint_data=torch.load(args.checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint_data["model"])

    text=generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(text)


if __name__ == "__main__":
    main()
