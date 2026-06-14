"""
GoogLeNet 教学版实现。

这个版本的重点是把 Inception 模块讲明白：同一层里并行走多条分支，
再把结果拼起来，让网络自己决定该更关注哪种感受野。
"""

from __future__ import annotations

import torch
from torch import nn

from model_blocks import ConvBNAct


class InceptionBlock(nn.Module):
    """教学版 Inception 模块。"""

    def __init__(self, in_channels: int, branch_channels: int):
        super().__init__()
        self.branch1 = ConvBNAct(in_channels, branch_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, padding=1),
        )
        self.branch5 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(in_channels, branch_channels, kernel_size=1),
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)]
        return torch.cat(outputs, dim=1)


class GoogLeNetModel(nn.Module):
    """教学版 GoogLeNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(32, 64, kernel_size=3, padding=1),
        )
        self.inception1 = InceptionBlock(64, 32)
        self.transition = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(128, 128, kernel_size=1),
        )
        self.inception2 = InceptionBlock(128, 48)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.transition(x)
        x = self.inception2(x)
        return self.head(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return GoogLeNetModel(num_classes=num_classes)
