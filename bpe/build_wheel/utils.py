"""
这个文件放 BPE 项目里会反复用到的展示和格式化函数。

教学项目里，输出怎么展示很重要，因为读者是否能看懂，
往往不只取决于算法正不正确，还取决于每一步有没有被清楚地打印出来。
"""

from __future__ import annotations


def print_section(title: str) -> None:
    """打印清晰的分段标题。"""
    line = "=" * 16
    print(f"\n{line} {title} {line}")


def format_symbol_sequence(symbols: list[str]) -> str:
    """把一个 token 序列格式化成更易读的字符串。"""
    return " ".join(symbols)


def format_word_state(word_state: dict[str, list[str]]) -> str:
    """把当前每个词的符号表示整理成多行文本。"""
    lines = []
    for word, symbols in word_state.items():
        lines.append(f"- {word:<12} -> {format_symbol_sequence(symbols)}")
    return "\n".join(lines)


def format_frequency_dict(freq_dict: dict, top_k: int | None = None) -> str:
    """把频率字典按频次从高到低展示。"""
    items = sorted(freq_dict.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        items = items[:top_k]
    return "\n".join(f"- {key}: {value}" for key, value in items)


def merge_pair_to_text(pair: tuple[str, str]) -> str:
    """把二元 pair 转成直观文本。"""
    return f"({pair[0]}, {pair[1]})"


def flatten(list_of_lists: list[list[str]]) -> list[str]:
    """把二维列表拍平。"""
    flattened = []
    for items in list_of_lists:
        flattened.extend(items)
    return flattened
