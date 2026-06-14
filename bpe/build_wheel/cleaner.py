"""
这个文件负责 BPE 项目的文本清洗。

这里允许使用 `re`，因为基础清洗不是本项目真正要复现的核心。
真正要教明白的重点，是后面的字符级表示、pair 统计和 merge 过程。
"""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """把原始文本清洗成更适合教学版 BPE 处理的形式。"""
    text = text.lower()

    # 只保留字母、数字和基础空白，去掉大部分标点，减少无教学价值的噪音。
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 连续空白压成一个空格，便于后续按词切分。
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_words(cleaned_text: str) -> list[str]:
    """把清洗后的文本按空格切成词。"""
    if not cleaned_text:
        return []
    return cleaned_text.split(" ")
