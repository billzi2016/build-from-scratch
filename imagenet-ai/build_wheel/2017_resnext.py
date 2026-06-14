"""
ResNeXt 教学版实现。

ResNeXt 的重点不是一味加深或加宽，而是在 block 中引入 cardinality，
也就是更强调“并行组数”这个维度。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, ResNeXtBlock


class ResNeXtModel(nn.Module):
    """教学版 ResNeXt。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResNeXtBlock(32, 64, groups=4),
            ResNeXtBlock(64, 64, groups=4),
        )
        self.layer2 = nn.Sequential(
            ResNeXtBlock(64, 128, stride=2, groups=4),
            ResNeXtBlock(128, 128, groups=4),
        )
        self.layer3 = nn.Sequential(
            ResNeXtBlock(128, 256, stride=2, groups=8),
            ResNeXtBlock(256, 256, groups=8),
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
    return ResNeXtModel(num_classes=num_classes)
