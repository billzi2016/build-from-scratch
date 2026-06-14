"""
MobileNet 教学版实现。

MobileNet 的关键是把标准卷积拆成 depthwise + pointwise 两步，
从而让初学者看到“轻量化”到底轻在什么地方。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, DepthwiseSeparableConv


class MobileNetModel(nn.Module):
    """教学版 MobileNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return MobileNetModel(num_classes=num_classes)
