"""
ShuffleNet 教学版实现。

这个版本重点展示两件事：分组卷积可以省计算，但会让通道彼此隔离；
于是需要 channel shuffle 把不同组的信息重新打散再混合。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, ShuffleUnit


class ShuffleNetModel(nn.Module):
    """教学版 ShuffleNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 24, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ShuffleUnit(24, 48, stride=2, groups=2),
            ShuffleUnit(48, 48, stride=1, groups=2),
        )
        self.stage3 = nn.Sequential(
            ShuffleUnit(48, 96, stride=2, groups=2),
            ShuffleUnit(96, 96, stride=1, groups=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return ShuffleNetModel(num_classes=num_classes)
