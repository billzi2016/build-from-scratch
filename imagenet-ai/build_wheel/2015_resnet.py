"""
ResNet 教学版实现。

这个版本重点展示残差连接：主分支学增量，shortcut 保留原始信息，
让更深的网络更容易训练，也让读者能直观看到“跳连”是怎么接的。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, ResidualBasicBlock


class ResNetModel(nn.Module):
    """教学版 ResNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBNAct(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResidualBasicBlock(64, 64),
            ResidualBasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResidualBasicBlock(64, 128, stride=2),
            ResidualBasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResidualBasicBlock(128, 256, stride=2),
            ResidualBasicBlock(256, 256),
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
    return ResNetModel(num_classes=num_classes)
