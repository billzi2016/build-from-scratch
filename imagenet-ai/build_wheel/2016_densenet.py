"""
DenseNet 教学版实现。

DenseNet 的教学重点是“后面每层都能直接拿到前面所有层的输出”，
所以这里会尽量把通道拼接过程写得直白。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct, DenseLayer, TransitionLayer


class DenseBlock(nn.Module):
    """由多个 dense layer 组成的块。"""

    def __init__(self, in_channels: int, growth_rate: int, layer_count: int):
        super().__init__()
        layers = []
        current_channels = in_channels
        for _ in range(layer_count):
            layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x):
        return self.block(x)


class DenseNetModel(nn.Module):
    """教学版 DenseNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block1 = DenseBlock(32, growth_rate=16, layer_count=3)
        self.transition1 = TransitionLayer(self.block1.out_channels, 48)
        self.block2 = DenseBlock(48, growth_rate=16, layer_count=3)
        self.transition2 = TransitionLayer(self.block2.out_channels, 64)
        self.block3 = DenseBlock(64, growth_rate=16, layer_count=3)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(self.block3.out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.block3.out_channels, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        x = self.transition2(x)
        x = self.block3(x)
        return self.classifier(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return DenseNetModel(num_classes=num_classes)
