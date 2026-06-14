"""
VGG 教学版实现。

VGG 的核心不是花哨，而是规整：大量 3x3 卷积重复堆叠，
让读者很容易看清“更深的网络是怎么一层层搭起来的”。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvBNAct


def make_vgg_stage(in_channels: int, out_channels: int, block_count: int) -> nn.Sequential:
    """构建 VGG 的一个卷积阶段。"""
    layers = []
    current_channels = in_channels
    for _ in range(block_count):
        layers.append(ConvBNAct(current_channels, out_channels, kernel_size=3, padding=1))
        current_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGGModel(nn.Module):
    """教学版 VGG。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_stage(3, 32, block_count=2),
            make_vgg_stage(32, 64, block_count=2),
            make_vgg_stage(64, 128, block_count=3),
            make_vgg_stage(128, 256, block_count=3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return VGGModel(num_classes=num_classes)
