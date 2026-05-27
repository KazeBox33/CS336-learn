from pathlib import Path
import regex as re
from collections import Counter, defaultdict

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
    pair_counts, pair_to_pretokens = build_pair_stats(pretoken_counts) # 记录了pair 频次 和  pair 到 pretokens的 路由

    while len(vocab) <vocab_size:
        if not pair_counts:
            break
        best_pair=max(pair_counts,key=lambda pair:(pair_counts[pair],pair))
        merges.append(best_pair)
        vocab[len(vocab)]=best_pair[0]+best_pair[1]

        affected_pretokens=list(pair_to_pretokens[best_pair])

        for old_pretoken in affected_pretokens:
            if old_pretoken not in pretoken_counts:
                continue

            freq=pretoken_counts.pop(old_pretoken) #  从频率中删除old_pretoken

            for old_pair in iter_pairs(old_pretoken):
                pair_counts[old_pair] -= freq #pair 频率中删除 old_pretoken 的pair
                if pair_counts[old_pair] <= 0: 
                    del pair_counts[old_pair]
                pair_to_pretokens[old_pair].discard(old_pretoken) #删除 old_pair 到 old_pretoken的路径
                # 下面是把新的加进去
            new_pretoken=merge_one_pretoken(old_pretoken,best_pair)
            pretoken_counts[new_pretoken] +=freq 

            for new_pair in iter_pairs(new_pretoken):
                pair_counts[new_pair] +=freq
                pair_to_pretokens[new_pair].add(new_pretoken)
  

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
      new_pretoken=merge_one_pretoken(pretoken,pair_to_merge)
      new_counts[new_pretoken] +=freq

    return new_counts


def iter_pairs(pretoken):
    for i in range(len(pretoken)-1):
        yield (pretoken[i],pretoken[i+1])

def build_pair_stats(pretoken_counts):
    pair_counts=Counter()
    pair_to_pretokens=defaultdict(set)

    for pretoken,freq in pretoken_counts.items():
        for pair in iter_pairs(pretoken):
            pair_counts[pair] +=freq  # 记录相邻token的频次
            pair_to_pretokens[pair].add(pretoken)  # 记录 token_pair 到 pretoken 的索引
    
    return pair_counts , pair_to_pretokens


def merge_one_pretoken(pretoken,pair_to_merge):
    merged_tokens=[]
    i = 0

    while i< len(pretoken):
        if i < len(pretoken) -1 and (pretoken[i],pretoken[i+1]) == pair_to_merge:
            merged_tokens.append(pretoken[i]+ pretoken[i+1])
            i+= 2
        else :
            merged_tokens.append(pretoken[i])
            i+=1

    return tuple(merged_tokens)
