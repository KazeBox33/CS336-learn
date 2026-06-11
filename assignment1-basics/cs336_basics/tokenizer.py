from __future__ import annotations
import regex as re
from collections.abc import Iterable, Iterator



PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
            self,
            vocab:dict[int,bytes],
            merges:list[tuple[bytes,bytes]],
            special_tokens:list[str]|None=None
    ):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens or []

        self.token_to_id={token:token_id for token_id , token in vocab.items()} #词表对应的是 token 到 token_id 的映射
        self.merge_ranks={pair:rank for rank, pair in enumerate(merges)}  # pair 对应 下标 , 把merge转换成一个快速查询表   融合的顺序也要遵照原本的训练时候的融合顺序

        if self.special_tokens:
            self.special_tokens=sorted(self.special_tokens,key=len,reverse=True) #从大到小排序 reverse=true 就说明从大到小排序
            self.special_pattern=re.compile("|".join(re.escape(token) for token in self.special_tokens)) # 要 有限匹配长的
        else :
            self.special_pattern = None

    
    def encode(self,text:str) -> list[int]:
        ids=[]

        for piece, is_special in self._split_on_special_tokens(text):
            if is_special:
                ids.append(self.token_to_id[piece.encode("utf-8")])
                continue

            for match in re.finditer(PAT,piece):
                pretoken=match.group().encode("utf-8")
                ids.extend(self._encode_pretoken(pretoken))
        
        return ids

    def _split_on_special_tokens(self,text:str): # 从文本中把  special token 和 普通的字符串切分出来
        if self.special_pattern is None:
            yield text, False
            return 
        
        start =0
        for match in self.special_pattern.finditer(text):
            if match.start()>start:
                yield text[start:match.start()],False

            yield match.group(),True
            start = match.end()

        if start<len(text):
            yield text[start:] , False

    def decode(self,ids:list[int]) -> str:
        token_bytes=b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8",errors="replace")
    

    def _encode_pretoken(self,pretoken:bytes) -> list[int]: #处理一个pre-token 应用BPE merges    找出最好的pair
        tokens=tuple(bytes([byte]) for byte in pretoken)

        while len(tokens) > 1 :
            best_pair=None
            best_rank=None

            for i in range(len(tokens)-1):
                pair=(tokens[i],tokens[i+1])
                rank=self.merge_ranks.get(pair)   

                if rank is not None and (best_rank is None or rank< best_rank):
                    best_pair=pair
                    best_rank=rank

            if best_pair is None: #表示没有可以继续拼的了
                break

            tokens = self._merge_tokens(tokens, best_pair) # 给 pair 合成到 token 当中

        return [self.token_to_id[token] for token in tokens]

    def _merge_tokens(self,tokens:tuple[bytes, ...],pair_to_merge:tuple[bytes,bytes]) ->tuple[bytes,...]:
        merged_tokens=[]
        i=0

        while i<len(tokens):
            if i <len(tokens)-1 and (tokens[i],tokens[i+1]) == pair_to_merge:
                merged_tokens.append(tokens[i]+tokens[i+1])
                i+=2
            else :
                merged_tokens.append(tokens[i])
                i +=1
        return tuple(merged_tokens)

    
    def encode_iterable(self,iterable:Iterable[str]) ->Iterator[int]:
        for text in iterable:
            yield from self.encode(text)  #encode生成的是list ， 这里yield from 直接一个个取出来，更加省内存

