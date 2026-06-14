"""
这个文件实现 BPE 训练主流程。

这里的重点是把 BPE 的核心算法一步一步写出来：
1. 从字符级表示开始
2. 统计相邻 pair 频率
3. 选出当前最常见的 pair
4. 把这个 pair 合并成一个新 token
5. 更新所有词的表示
"""

from __future__ import annotations

from dataclasses import dataclass

from vocab import build_initial_word_state, build_vocabulary_from_state


@dataclass
class MergeStep:
    """记录每一轮 merge 的关键信息，方便在 main.py 里完整展示。"""

    step_index: int
    best_pair: tuple[str, str]
    pair_frequency: int
    merged_token: str
    vocabulary: list[str]


@dataclass
class BPETrainingResult:
    """汇总训练后的所有结果。"""

    word_frequency: dict[str, int]
    initial_word_state: dict[str, list[str]]
    final_word_state: dict[str, list[str]]
    merges: list[tuple[str, str]]
    merge_steps: list[MergeStep]
    vocabulary: list[str]


def count_pair_frequencies(
    word_state: dict[str, list[str]],
    word_frequency: dict[str, int],
) -> dict[tuple[str, str], int]:
    """统计当前所有相邻 pair 的加权频率。"""
    pair_frequency: dict[tuple[str, str], int] = {}

    for word, symbols in word_state.items():
        frequency = word_frequency[word]
        for index in range(len(symbols) - 1):
            pair = (symbols[index], symbols[index + 1])
            pair_frequency[pair] = pair_frequency.get(pair, 0) + frequency

    return pair_frequency


def merge_symbols(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """
    在一个词的当前符号序列中执行一次 pair merge。

    这里不用任何花哨写法，保持一步一步扫描，
    让读者清楚看到 pair 是如何变成新 token 的。
    """
    merged = []
    index = 0
    merged_token = pair[0] + pair[1]

    while index < len(symbols):
        if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
            merged.append(merged_token)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1

    return merged


def apply_merge_to_state(
    word_state: dict[str, list[str]],
    pair: tuple[str, str],
) -> dict[str, list[str]]:
    """把一个 merge 应用到全部词的当前表示上。"""
    updated_state = {}
    for word, symbols in word_state.items():
        updated_state[word] = merge_symbols(symbols, pair)
    return updated_state


def choose_best_pair(pair_frequency: dict[tuple[str, str], int]) -> tuple[tuple[str, str], int] | None:
    """选出当前频率最高的 pair。"""
    if not pair_frequency:
        return None

    best_pair = None
    best_frequency = -1
    for pair, frequency in pair_frequency.items():
        if frequency > best_frequency:
            best_pair = pair
            best_frequency = frequency

    if best_pair is None:
        return None
    return best_pair, best_frequency


class BPETrainer:
    """教学版 BPE 训练器。"""

    def __init__(self, max_merges: int = 10):
        self.max_merges = max_merges

    def train(self, word_frequency: dict[str, int]) -> BPETrainingResult:
        """根据词频训练 merge 规则。"""
        initial_word_state = build_initial_word_state(word_frequency)
        current_state = {word: symbols[:] for word, symbols in initial_word_state.items()}
        merge_steps: list[MergeStep] = []
        merges: list[tuple[str, str]] = []

        for step_index in range(1, self.max_merges + 1):
            pair_frequency = count_pair_frequencies(current_state, word_frequency)
            best_choice = choose_best_pair(pair_frequency)
            if best_choice is None:
                break

            best_pair, best_frequency = best_choice
            if best_frequency <= 1:
                break

            current_state = apply_merge_to_state(current_state, best_pair)
            merges.append(best_pair)

            vocabulary = build_vocabulary_from_state(current_state)
            merge_steps.append(
                MergeStep(
                    step_index=step_index,
                    best_pair=best_pair,
                    pair_frequency=best_frequency,
                    merged_token=best_pair[0] + best_pair[1],
                    vocabulary=vocabulary,
                )
            )

        final_vocabulary = build_vocabulary_from_state(current_state)

        return BPETrainingResult(
            word_frequency=dict(word_frequency),
            initial_word_state=initial_word_state,
            final_word_state=current_state,
            merges=merges,
            merge_steps=merge_steps,
            vocabulary=final_vocabulary,
        )
