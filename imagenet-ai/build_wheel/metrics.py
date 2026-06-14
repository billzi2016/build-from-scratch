"""
这个文件实现训练和测试阶段会用到的评价指标。

为了减少外部依赖，这里不用 sklearn，而是直接用 torch
手写 ACC、macro F1 和多分类 one-vs-rest macro AUC。
"""

from __future__ import annotations

import torch


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率。"""
    return (predictions == targets).float().mean().item()


def compute_macro_f1(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """手写多分类 macro F1。"""
    f1_scores = []
    for class_id in range(num_classes):
        predicted_positive = predictions == class_id
        actual_positive = targets == class_id

        true_positive = (predicted_positive & actual_positive).sum().item()
        false_positive = (predicted_positive & (~actual_positive)).sum().item()
        false_negative = ((~predicted_positive) & actual_positive).sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0

        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))

    return float(sum(f1_scores) / len(f1_scores))


def _compute_average_ranks(scores: torch.Tensor) -> torch.Tensor:
    """
    给分数计算平均秩。

    这里专门处理并列分数，保证 AUC 的实现更稳妥。
    """
    sorted_scores, sorted_indices = torch.sort(scores)
    ranks = torch.zeros_like(sorted_scores, dtype=torch.float32)

    start = 0
    total_count = sorted_scores.numel()
    while start < total_count:
        end = start + 1
        while end < total_count and sorted_scores[end].item() == sorted_scores[start].item():
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end

    original_order_ranks = torch.zeros_like(ranks)
    original_order_ranks[sorted_indices] = ranks
    return original_order_ranks


def _binary_auc(scores: torch.Tensor, binary_targets: torch.Tensor) -> float | None:
    """用秩和公式计算二分类 AUC。"""
    binary_targets = binary_targets.to(torch.long)
    positive_count = int((binary_targets == 1).sum().item())
    negative_count = int((binary_targets == 0).sum().item())

    if positive_count == 0 or negative_count == 0:
        return None

    ranks = _compute_average_ranks(scores)
    positive_rank_sum = ranks[binary_targets == 1].sum().item()
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def compute_macro_ovr_auc(probabilities: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """计算多分类 one-vs-rest macro AUC。"""
    auc_values = []
    for class_id in range(num_classes):
        class_scores = probabilities[:, class_id]
        binary_targets = (targets == class_id).to(torch.long)
        class_auc = _binary_auc(class_scores, binary_targets)
        if class_auc is not None:
            auc_values.append(class_auc)

    if not auc_values:
        return 0.0
    return float(sum(auc_values) / len(auc_values))
