"""
SENet 教学版实现。

SENet 很适合教学，因为它展示了一种非常直观的想法：
网络不仅提特征，还能学会“哪些通道更重要”。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, SEResidualBlock


class SENetModel(nn.Module):
    """教学版 SENet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            SEResidualBlock(32, 64),
            SEResidualBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            SEResidualBlock(64, 128, stride=2),
            SEResidualBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            SEResidualBlock(128, 256, stride=2),
            SEResidualBlock(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return SENetModel(num_classes=num_classes)
