from __future__ import annotations
import regex as re

class Tokenizer:
    def __init__(
            self,
            vocab:dict[int,bytes],
            merges:list[tuple[bytes,bytes]],
            special_tokens:list[str]|None=None
    ):
        self.vocab=vocab
        self.merges=merges
        self.specila_tokens=special_tokens or []

        self.token_to_id={token:token_id for token_id , token in vocab.items()} #词表对应的是 token 到 token_id 的映射
        self.merge_ranks={pair:rank for rank, pair in enumerate(merges)}  # pair 对应 下标 , 把merge转换成一个快速查询表

        if self.special_tokens:
            self.special_tokens=sorted(self.special_tokens,key=len,reverse=True) #从大到小排序 reverse=true 就说明从大到小排序
            self.special_pattern=re.compile("|".join(re.escape(token) for token in self.special_tokens)) # 要 有限匹配长的
        else :
            self.special_pattern = None

    
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
