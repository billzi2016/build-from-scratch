"""
EfficientNet 教学版实现。

这里不会把 EfficientNet 的所有工程细节都搬进来，而是抓住最关键的
 MBConv + SE + 更平衡的 stage 设计，让初学者先理解它为什么“高效”。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, MBConvBlock


class EfficientNetModel(nn.Module):
    """教学版 EfficientNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            MBConvBlock(32, 32, stride=1, expand_ratio=2),
            MBConvBlock(32, 48, stride=2, expand_ratio=4),
        )
        self.stage2 = nn.Sequential(
            MBConvBlock(48, 48, stride=1, expand_ratio=4),
            MBConvBlock(48, 64, stride=2, expand_ratio=4),
        )
        self.stage3 = nn.Sequential(
            MBConvBlock(64, 96, stride=1, expand_ratio=4),
            MBConvBlock(96, 96, stride=1, expand_ratio=4),
        )
        self.head = nn.Sequential(
            ConvBNAct(96, 128, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return EfficientNetModel(num_classes=num_classes)
