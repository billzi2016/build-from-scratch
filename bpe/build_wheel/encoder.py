"""
这个文件负责把文本按训练得到的 merge 规则编码成 token 序列。

教学上这里有个关键点：
训练学到的是 merge 顺序，所以编码时也要按同样顺序，
把每个词的字符级表示一步一步合并。
"""

from __future__ import annotations

from cleaner import normalize_text, split_words
from trainer import merge_symbols
from vocab import END_OF_WORD, build_token_to_id, word_to_symbols


class BPEEncoder:
    """教学版 BPE 编码器。"""

    def __init__(self, merges: list[tuple[str, str]], vocabulary: list[str]):
        self.merges = merges
        self.vocabulary = vocabulary
        self.token_to_id = build_token_to_id(vocabulary)

    def encode_word(self, word: str) -> list[str]:
        """把单个词编码成 token 序列。"""
        symbols = word_to_symbols(word)
        for pair in self.merges:
            symbols = merge_symbols(symbols, pair)
        return symbols

    def encode_text(self, text: str) -> dict:
        """把整段文本编码成 token 文本和 token id。"""
        cleaned_text = normalize_text(text)
        words = split_words(cleaned_text)

        encoded_words = [self.encode_word(word) for word in words]
        encoded_tokens = [token for tokens in encoded_words for token in tokens]
        encoded_ids = [self.token_to_id[token] for token in encoded_tokens]

        return {
            "cleaned_text": cleaned_text,
            "words": words,
            "encoded_words": encoded_words,
            "encoded_tokens": encoded_tokens,
            "encoded_ids": encoded_ids,
        }


def tokens_to_display_text(tokens: list[str]) -> str:
    """把 token 序列整理成更适合展示的字符串。"""
    return " | ".join(tokens).replace(END_OF_WORD, "</w>")
