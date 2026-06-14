"""
本文件是 imagenet-ai 项目的统一入口。

它负责做三件事：
1. 展示可选模型列表。
2. 动态加载某个年份模型文件。
3. 串联 train / test / early stopping / ACC / F1 / AUC 的完整流程。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_DIR / "imagenet"

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dataset import load_datasets
from model_registry import MODEL_REGISTRY, get_model_keys, get_model_spec
from trainer import TrainingConfig, create_data_loaders, train_and_evaluate
from utils import count_parameters, format_parameter_count, load_module_from_path, set_seed


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="imagenet-ai 教学版模型训练入口")
    parser.add_argument("--model", default="alexnet", help="模型 key，或使用 all 依次训练全部模型。")
    parser.add_argument("--epochs", type=int, default=12, help="最大训练轮数。")
    parser.add_argument("--patience", type=int, default=3, help="early stopping 的耐心值。")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减。")
    parser.add_argument("--image-size", type=int, default=64, help="教学版输入图像大小。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--list", action="store_true", help="只展示当前支持的模型列表。")
    return parser


def print_model_list() -> None:
    """打印项目支持的模型列表。"""
    print("当前支持的代表模型：")
    for item in MODEL_REGISTRY:
        print(
            f"- {item['year']} | {item['key']:<18} | "
            f"{item['display_name']:<20} | {item['summary']}"
        )


def load_model_builder(model_key: str):
    """根据模型 key 动态加载对应文件，并返回模块对象。"""
    spec = get_model_spec(model_key)
    module_path = CURRENT_DIR / spec["file_name"]
    module_name = f"imagenet_ai_{model_key}"
    return load_module_from_path(module_name, module_path), spec


def run_single_model(model_key: str, args) -> dict:
    """训练并测试单个模型。"""
    module, spec = load_model_builder(model_key)
    train_dataset, test_dataset = load_datasets(DATA_DIR, image_size=args.image_size)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.create_model(num_classes=train_dataset.num_classes, image_size=args.image_size).to(device)

    print("")
    print(f"开始训练: {spec['display_name']} ({spec['year']})")
    print(f"模型说明: {spec['summary']}")
    print(f"类别数: {train_dataset.num_classes}")
    print(f"参数量: {format_parameter_count(count_parameters(model))}")

    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )

    result = train_and_evaluate(model, train_loader, test_loader, device=device, config=config)
    final_test = result["final_test"]

    print("最终测试指标：")
    print(
        f"ACC={final_test['acc']:.4f} | "
        f"F1={final_test['f1']:.4f} | "
        f"AUC={final_test['auc']:.4f}"
    )

    return {
        "model_key": model_key,
        "display_name": spec["display_name"],
        "year": spec["year"],
        "metrics": final_test,
    }


def run_all_models(args) -> None:
    """按顺序训练所有模型，并汇总最终指标。"""
    summaries = []
    for model_key in get_model_keys():
        summaries.append(run_single_model(model_key, args))

    print("")
    print("全部模型最终测试汇总：")
    for item in summaries:
        metrics = item["metrics"]
        print(
            f"- {item['year']} {item['display_name']}: "
            f"ACC={metrics['acc']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"AUC={metrics['auc']:.4f}"
        )


def main() -> None:
    """程序入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    if args.list:
        print_model_list()
        return

    if args.model == "all":
        run_all_models(args)
        return

    if args.model not in get_model_keys():
        available = ", ".join(get_model_keys())
        raise ValueError(f"未知模型: {args.model}。可选值: {available}, all")

    run_single_model(args.model, args)


if __name__ == "__main__":
    main()
