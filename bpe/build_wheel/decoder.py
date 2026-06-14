"""
这个文件负责把 BPE token 序列解码回文本。

解码不是本项目最复杂的部分，但它对初学者很关键，
因为它能帮助读者确认：merge 学到的 token 最后还能重新还原出词。
"""

from __future__ import annotations

from vocab import END_OF_WORD


class BPEDecoder:
    """教学版 BPE 解码器。"""

    def decode_tokens(self, tokens: list[str]) -> str:
        """把 token 序列还原成文本。"""
        words = []
        current_word = ""

        for token in tokens:
            if token.endswith(END_OF_WORD):
                current_word += token[: -len(END_OF_WORD)]
                words.append(current_word)
                current_word = ""
            else:
                current_word += token

        if current_word:
            words.append(current_word)

        return " ".join(words)
