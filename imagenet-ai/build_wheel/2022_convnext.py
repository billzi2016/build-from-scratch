"""
ConvNeXt 教学版实现。

ConvNeXt 很适合放在时间线末尾，因为它能让读者看到：
即使 Transformer 很强，CNN 也可以吸收新时期设计习惯后重新变强。
"""

from __future__ import annotations

from torch import nn

from model_blocks import ConvNeXtBlock


class ConvNeXtModel(nn.Module):
    """教学版 ConvNeXt。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # stem 用较大的 patchify 卷积起步，是 ConvNeXt 常见风格。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=4, stride=4),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(48),
            ConvNeXtBlock(48),
        )
        self.downsample1 = nn.Conv2d(48, 96, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(96),
            ConvNeXtBlock(96),
        )
        self.downsample2 = nn.Conv2d(96, 192, kernel_size=2, stride=2)
        self.stage3 = nn.Sequential(
            ConvNeXtBlock(192),
            ConvNeXtBlock(192),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(192),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        return self.head(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    del image_size
    return ConvNeXtModel(num_classes=num_classes)
