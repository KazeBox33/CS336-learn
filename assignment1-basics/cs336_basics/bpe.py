from pathlib import Path
import os
import heapq
import regex as re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MIN_PARALLEL_FILE_BYTES = 1_000_000


class _MaxPair:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def push_pair(heap, pair_counts, pair):
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(heap, (-count, _MaxPair(pair), pair))


def pop_best_pair(heap, pair_counts):
    while heap:
        neg_count, _, pair = heapq.heappop(heap)
        count = -neg_count
        if pair_counts.get(pair, 0) == count:
            return pair
    return None


def train_bpe(
        input_path:str|Path,
        vocab_size:int,
        special_tokens:list[str],
        num_processes:int|None=None,
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    vocab = {}

    for i in range(256):
        vocab[i]=bytes([i])

    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")
    
    merges=[]

    pretoken_counts=pretokenize_file(input_path,special_tokens,num_processes) # pretoken_counts的key是 每个被预处理后的词分解后的tuple
    pair_counts, pair_to_pretokens = build_pair_stats(pretoken_counts) # 记录了pair 频次 和  pair 到 pretokens的 路由
    pair_heap=[]
    for pair in pair_counts:
        push_pair(pair_heap,pair_counts,pair)

    while len(vocab) <vocab_size:
        if not pair_counts:
            break
        best_pair=pop_best_pair(pair_heap,pair_counts)
        if best_pair is None:
            break
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
                else:
                    push_pair(pair_heap,pair_counts,old_pair)
                pair_to_pretokens[old_pair].discard(old_pretoken) #删除 old_pair 到 old_pretoken的路径
                # 下面是把新的加进去
            new_pretoken=merge_one_pretoken(old_pretoken,best_pair)
            pretoken_counts[new_pretoken] +=freq 

            for new_pair in iter_pairs(new_pretoken):
                pair_counts[new_pair] +=freq
                pair_to_pretokens[new_pair].add(new_pretoken)
                push_pair(pair_heap,pair_counts,new_pair)
  

    return vocab,merges


def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size=file.tell()
    file.seek(0)

    chunk_size=file_size // desired_num_chunks
    chunk_boundaries=[i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1]=file_size

    mini_chunk_size=4096

    for bi in range(1, len(chunk_boundaries)-1):
        initial_position=chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk=file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi]=file_size
                break

            found_at=mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi]=initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize_text(text,special_tokens):
    special_pattern="|".join(re.escape(token) for token in special_tokens)
    if special_tokens:
        segments=re.split(special_pattern,text)
    else:
        segments=[text]

    pretoken_counts=Counter()

    for segment in segments:
        for match in re.finditer(PAT,segment):
            pretoken=match.group()
            pretoken_bytes=pretoken.encode("utf-8")
            pretoken_tuple=tuple(bytes([b]) for b in pretoken_bytes)
            pretoken_counts[pretoken_tuple] +=1

    return pretoken_counts


def pretokenize_chunk(args):
    input_path,start,end,special_tokens=args
    with Path(input_path).open("rb") as file:
        file.seek(start)
        text=file.read(end-start).decode("utf-8",errors="ignore")
    return pretokenize_text(text,special_tokens)


def pretokenize_file(input_path,special_tokens,num_processes=None):
    input_path=Path(input_path)
    file_size=input_path.stat().st_size

    if num_processes is None:
        num_processes=min(os.cpu_count() or 1,8)

    if num_processes <= 1 or file_size < MIN_PARALLEL_FILE_BYTES or not special_tokens:
        text=input_path.read_text(encoding="utf-8")
        return pretokenize_text(text,special_tokens)

    split_special_token=special_tokens[0].encode("utf-8")
    with input_path.open("rb") as file:
        boundaries=find_chunk_boundaries(file,num_processes,split_special_token)

    chunk_args=[
        (input_path,start,end,special_tokens)
        for start,end in zip(boundaries[:-1],boundaries[1:])
        if end > start
    ]

    if len(chunk_args) <= 1:
        text=input_path.read_text(encoding="utf-8")
        return pretokenize_text(text,special_tokens)

    pretoken_counts=Counter()
    with ProcessPoolExecutor(max_workers=min(num_processes,len(chunk_args))) as executor:
        for chunk_counts in executor.map(pretokenize_chunk,chunk_args):
            pretoken_counts.update(chunk_counts)

    return pretoken_counts



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
