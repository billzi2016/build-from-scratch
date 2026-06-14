"""
这个文件负责 BPE 里的词表和初始符号表示。

教学版 BPE 这里采用最常见、最容易解释的一种做法：
把每个词拆成字符序列，并在结尾加一个词尾标记 `</w>`。
这样读者能看到：
1. 初始最小粒度是什么
2. merge 到底是如何发生在字符级表示上的
"""

from __future__ import annotations

from collections import Counter


END_OF_WORD = "</w>"


def build_word_frequency(words: list[str]) -> Counter:
    """统计清洗后语料里的词频。"""
    return Counter(words)


def word_to_symbols(word: str) -> list[str]:
    """把单词转成字符级初始表示，并在结尾加词尾标记。"""
    return list(word) + [END_OF_WORD]


def build_initial_word_state(word_frequency: Counter) -> dict[str, list[str]]:
    """为每个去重后的词建立当前符号表示。"""
    return {word: word_to_symbols(word) for word in word_frequency}


def build_vocabulary_from_state(word_state: dict[str, list[str]]) -> list[str]:
    """根据当前每个词的符号表示，整理出当前词表。"""
    vocab = sorted({symbol for symbols in word_state.values() for symbol in symbols})
    return vocab


def build_token_to_id(vocab: list[str]) -> dict[str, int]:
    """给当前词表分配稳定的 token id。"""
    return {token: index for index, token in enumerate(vocab)}
