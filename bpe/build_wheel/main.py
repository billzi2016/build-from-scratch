"""
这个文件是 BPE 教学版项目的总入口。

运行它时，读者应该能一次看到：
1. 原始文本
2. 清洗后的文本
3. 词频统计
4. 初始字符级表示
5. 每轮 merge 的最佳 pair
6. 最终词表
7. 示例文本 encode / decode 的结果
"""

from __future__ import annotations

import argparse

from cleaner import normalize_text, split_words
from decoder import BPEDecoder
from encoder import BPEEncoder, tokens_to_display_text
from trainer import BPETrainer
from utils import (
    format_frequency_dict,
    format_symbol_sequence,
    format_word_state,
    merge_pair_to_text,
    print_section,
)
from vocab import build_word_frequency


DEFAULT_TRAINING_TEXT = """
low lower lowest low lower
new newer newest new new
wide wider widest wide
slow slower slowest slow
"""

DEFAULT_EXAMPLE_TEXT = "lower newest slowest"


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="教学版 BPE from scratch 演示入口")
    parser.add_argument("--merges", type=int, default=10, help="最多执行多少轮 merge。")
    parser.add_argument("--text", type=str, default=DEFAULT_TRAINING_TEXT, help="训练语料文本。")
    parser.add_argument("--example", type=str, default=DEFAULT_EXAMPLE_TEXT, help="用于演示编码和解码的示例文本。")
    return parser


def print_training_overview(raw_text: str, cleaned_text: str, word_frequency: dict[str, int], initial_word_state: dict[str, list[str]]) -> None:
    """打印训练开始前的关键信息。"""
    print_section("原始文本")
    print(raw_text.strip())

    print_section("清洗后文本")
    print(cleaned_text)

    print_section("词频统计")
    print(format_frequency_dict(word_frequency))

    print_section("初始字符级表示")
    print(format_word_state(initial_word_state))


def print_merge_process(training_result) -> None:
    """逐轮打印 merge 过程。"""
    print_section("Merge 过程")

    if not training_result.merge_steps:
        print("当前语料没有足够高频的 pair，可用于进一步 merge。")
        return

    for step in training_result.merge_steps:
        print(f"第 {step.step_index} 轮 merge")
        print(f"- 最佳 pair: {merge_pair_to_text(step.best_pair)}")
        print(f"- pair 频率: {step.pair_frequency}")
        print(f"- 合并结果: {step.merged_token}")
        print(f"- 当前词表大小: {len(step.vocabulary)}")
        print("")


def print_final_result(training_result) -> None:
    """打印训练完成后的结果。"""
    print_section("最终词的表示")
    print(format_word_state(training_result.final_word_state))

    print_section("最终 merges")
    if training_result.merges:
        for index, pair in enumerate(training_result.merges, start=1):
            print(f"- 第 {index} 条: {merge_pair_to_text(pair)} -> {pair[0] + pair[1]}")
    else:
        print("没有学到新的 merge。")

    print_section("最终词表")
    for token in training_result.vocabulary:
        print(f"- {token}")


def print_encoding_demo(example_text: str, encoder: BPEEncoder, decoder: BPEDecoder) -> None:
    """展示示例文本的编码和解码过程。"""
    encoded = encoder.encode_text(example_text)
    decoded_text = decoder.decode_tokens(encoded["encoded_tokens"])

    print_section("编码演示")
    print(f"示例文本: {example_text}")
    print(f"清洗后:   {encoded['cleaned_text']}")
    print(f"按词切分: {encoded['words']}")

    print("\n每个词的编码结果：")
    for word, encoded_word in zip(encoded["words"], encoded["encoded_words"]):
        print(f"- {word:<12} -> {format_symbol_sequence(encoded_word)}")

    print("\n完整 token 序列：")
    print(tokens_to_display_text(encoded["encoded_tokens"]))

    print("\n对应 token id：")
    print(encoded["encoded_ids"])

    print("\n解码结果：")
    print(decoded_text)


def main() -> None:
    """程序入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    cleaned_text = normalize_text(args.text)
    words = split_words(cleaned_text)
    word_frequency = build_word_frequency(words)

    trainer = BPETrainer(max_merges=args.merges)
    training_result = trainer.train(word_frequency)

    print_training_overview(
        raw_text=args.text,
        cleaned_text=cleaned_text,
        word_frequency=training_result.word_frequency,
        initial_word_state=training_result.initial_word_state,
    )
    print_merge_process(training_result)
    print_final_result(training_result)

    encoder = BPEEncoder(training_result.merges, training_result.vocabulary)
    decoder = BPEDecoder()
    print_encoding_demo(args.example, encoder, decoder)


if __name__ == "__main__":
    main()
