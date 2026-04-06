import regex as re
import json
from typing import Iterator, Iterable


class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens

        self.vocab2id={v: k for k, v in self.vocab.items()}
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,"r",encoding="utf-8") as f:
            vocab = json.load(f)
        # vocab = {k:v for k, v in vocab.items()}
        
        with open(merges_filepath,"r",encoding="utf-8") as f:
            lines = f.readlines()
        merges = [line.strip().split(" ") for line in lines]
        merges = [tuple(bytes(a), bytes(b)) for a,b in merges]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        text_chunks = self.split_text(text, self.special_tokens, drop_special=False)   #保留特殊token
        ids = []
        for text_chunk in text_chunks:
            if self.special_tokens and text_chunk in self.special_tokens:
                ids.append(self.vocab2id[text_chunk.encode("utf-8")])
            else:
                ids.extend(self.apply_merge(text_chunk))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        part_bytes = [self.vocab[idx] for idx in ids]
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors = "replace")

    def split_text(self,text, special_tokens = None, drop_special = True) -> list[str]:
        if special_tokens == None:
            return [text]
        special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
        pattern = "|".join(re.escape(special_token) for special_token in special_tokens)
        if not drop_special:
            pattern = f"({pattern})"
        compile_pattern = re.compile(pattern)
        text = compile_pattern.split(text)
        text = [c for c in text if c]    # remove empty strings
        return text
    
    def word2bytes(self,word):
        ids = list(word.encode("utf-8"))
        return [bytes([idx]) for idx in ids]
    
    def merge(self, word):
        bytes_word = self.word2bytes(word)
    
        while len(bytes_word) > 1:
            min_token_id = float('inf')
            best_pair_idx = -1
            merged_bytes = None
        
            # 查找最优合并对
            for i in range(len(bytes_word) - 1):
                pair = (bytes_word[i], bytes_word[i+1])
                if pair in self.merges:
                    combined = bytes_word[i] + bytes_word[i+1]
                    token_id = self.vocab2id.get(combined)
                    if token_id is not None and token_id < min_token_id:
                        min_token_id = token_id
                        best_pair_idx = i
                        merged_bytes = combined
        
            # 如果没有可合并的对，退出循环
            if best_pair_idx == -1:
                break
            
            # 执行最优合并
            bytes_word = (
                bytes_word[:best_pair_idx] + 
                [merged_bytes] + 
                bytes_word[best_pair_idx + 2:]
            )
    
        return [self.vocab2id[byte] for byte in bytes_word]

    def apply_merge(self,text_chunk):
        words = re.findall(self.compiled_pattern, text_chunk)
        ids=[]
        for word in words:
            ids.extend(self.merge(word))
        return ids
