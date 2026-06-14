"""
AlexNet 教学版实现。

这个版本的重点，不是逐像素复刻论文超参数，而是用 torch
把 AlexNet 的主干思路完整搭出来：大卷积起步、逐层提特征、
池化降采样、最后接全连接分类头。
"""

from __future__ import annotations

from torch import nn


class AlexNetModel(nn.Module):
    """教学版 AlexNet。"""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # 开头几层使用较大的卷积核和较快的下采样，体现 AlexNet 的时代风格。
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def create_model(num_classes: int = 3, image_size: int = 64):
    """统一导出模型构造函数。"""
    del image_size
    return AlexNetModel(num_classes=num_classes)
