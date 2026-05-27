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
    special_pattern="|".join(re.escape(token) for token in special_tokens) #python 的生成器

    segments=[]
    if special_tokens:
        segments=re.split(special_pattern,text) # 按照special_tokens 切出来的文本
    else:
        segments=[text]
    
    pretoken_counts=Counter()

    for segment in segments: #遍历每一段内容
        for match in re.finditer(PAT,segment):
            pretoken=match.group()
            pretoken_bytes=pretoken.encode("utf-8")
            pretoken_tuple=tuple(bytes([b]) for b in pretoken_bytes)
            pretoken_counts[pretoken_tuple] +=1
    while len(vocab) <vocab_size:
        pair_counts=count_pairs(pretoken_counts)

        if not pair_counts:
            break
        best_pair=max(pair_counts,key=lambda pair:(pair_counts[pair],pair))
        merges.append(best_pair)
        vocab[len(vocab)]=best_pair[0]+best_pair[1]
        pretoken_counts=merge_pretokens(pretoken_counts,best_pair)  # 获得了新的 pre_token 的计数表
  

    return vocab,merges



def count_pairs(pretoken_counts) ->Counter: # 
    pair_counts=Counter()
    
    for pretoken,freq in pretoken_counts.items():
        if len(pretoken) <2:
            continue

        for i in range(len(pretoken)-1):
            pair=(pretoken[i],pretoken[i+1])
            pair_counts[pair] += freq
    return pair_counts

def merge_pretokens(pretoken_counts,pair_to_merge):
    new_counts=Counter()

    for pretoken,freq in pretoken_counts.items():
        merged_tokens=[]
        i=0

        while i < len(pretoken):
            if i < len(pretoken) -1 and (pretoken[i],pretoken[i+1]) == pair_to_merge :
                merged_tokens.append(pretoken[i]+pretoken[i+1])
                i+=2
            else:
                merged_tokens.append(pretoken[i])
                i+=1
            
        new_pretoken=tuple(merged_tokens)
        new_counts[new_pretoken] +=freq

    return new_counts



