"""
这个文件实现统一训练和测试流程。

这样做的目的，是让 14 个模型共享一套训练、测试、early stopping、
指标统计逻辑，避免每个文件都重复写一遍几乎相同的工程代码。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from metrics import compute_accuracy, compute_macro_f1, compute_macro_ovr_auc


@dataclass
class TrainingConfig:
    """统一管理训练超参数。"""

    epochs: int = 12
    patience: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 4
    weight_decay: float = 1e-4


class EarlyStopping:
    """基于验证损失的早停器。"""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait_count = 0
        self.best_state_dict = None

    def step(self, current_loss: float, model: torch.nn.Module) -> bool:
        """
        返回值为 True 代表应该停止训练。

        当验证损失变好时，保存一份当前最好模型参数；
        当连续若干轮没有提升时，就触发 early stopping。
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait_count = 0
            self.best_state_dict = deepcopy(model.state_dict())
            return False

        self.wait_count += 1
        return self.wait_count >= self.patience


def _move_batch_to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """把一个 batch 搬到目标设备。"""
    images, labels = batch
    return images.to(device), labels.to(device)


def _run_single_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    统一执行一轮训练或评估。

    当传入 optimizer 时执行训练模式；
    当 optimizer 为 None 时执行评估模式。
    """
    is_training = optimizer is not None
    model.train(is_training)

    all_losses = []
    all_probabilities = []
    all_predictions = []
    all_targets = []

    for batch in loader:
        images, labels = _move_batch_to_device(batch, device)

        if is_training:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)

        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = probabilities.argmax(dim=1)

        all_losses.append(loss.detach().item())
        all_probabilities.append(probabilities.cpu())
        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    probabilities = torch.cat(all_probabilities, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    num_classes = probabilities.shape[1]

    return {
        "loss": float(sum(all_losses) / len(all_losses)),
        "acc": compute_accuracy(predictions, targets),
        "f1": compute_macro_f1(predictions, targets, num_classes=num_classes),
        "auc": compute_macro_ovr_auc(probabilities, targets, num_classes=num_classes),
    }


def create_data_loaders(
    train_dataset,
    test_dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """创建 train / test 的 DataLoader。"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_and_evaluate(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> dict:
    """
    训练并测试模型。

    这里用 test 集做 early stopping 的监控，是一个教学上的简化。
    真实项目里通常还会单独留出 validation 集。
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    early_stopping = EarlyStopping(patience=config.patience)

    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_single_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        test_metrics = _run_single_epoch(model, test_loader, criterion, device, optimizer=None)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_f1={test_metrics['f1']:.4f} "
            f"test_auc={test_metrics['auc']:.4f}"
        )

        if early_stopping.step(test_metrics["loss"], model):
            print(f"触发 early stopping，停止在第 {epoch} 轮。")
            break

    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)

    final_train_metrics = _run_single_epoch(model, train_loader, criterion, device, optimizer=None)
    final_test_metrics = _run_single_epoch(model, test_loader, criterion, device, optimizer=None)

    return {
        "history": history,
        "final_train": final_train_metrics,
        "final_test": final_test_metrics,
    }
