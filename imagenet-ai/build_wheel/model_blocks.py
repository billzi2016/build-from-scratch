"""
这个文件放多个模型都会复用的基础积木。

这里的复用不是“导入现成 ResNet/ViT 成品”，而是把卷积块、
残差块、SE、MBConv、Patch Embedding、Transformer Block
这种最小教学积木统一写好，后续每个模型再自己拼结构。
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    """卷积 + BN + 激活的常见组合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation if activation is not None else nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBasicBlock(nn.Module):
    """教学版 ResNet 基础残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.activation(out)


class ResNeXtBlock(nn.Module):
    """教学版 ResNeXt 分组残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 4):
        super().__init__()
        bottleneck_channels = out_channels // 2
        self.reduce = ConvBNAct(in_channels, bottleneck_channels, kernel_size=1)
        self.group_conv = ConvBNAct(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
        )
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.reduce(x)
        out = self.group_conv(out)
        out = self.expand(out)
        out = out + identity
        return self.activation(out)


class SEBlock(nn.Module):
    """SENet 的 squeeze-and-excitation 模块。"""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        pooled = self.pool(x).view(batch_size, channels)
        scale = self.fc(pooled).view(batch_size, channels, 1, 1)
        return x * scale


class SEResidualBlock(nn.Module):
    """带 SE 的残差块。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        return self.activation(out)


class DepthwiseSeparableConv(nn.Module):
    """MobileNet 的深度可分离卷积。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """ShuffleNet 的通道混洗操作。"""
    batch_size, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, height, width)


class ShuffleUnit(nn.Module):
    """教学版 ShuffleNet 单元。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 2):
        super().__init__()
        mid_channels = out_channels // 2
        self.stride = stride
        self.groups = groups

        self.branch = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=1, groups=groups),
            ConvBNAct(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=mid_channels,
            ),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch(x)
        out = channel_shuffle(out, self.groups)
        out = out + self.shortcut(x)
        return self.activation(out)


class DenseLayer(nn.Module):
    """DenseNet 里的单个 dense layer。"""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class TransitionLayer(nn.Module):
    """DenseNet block 之间的过渡层。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MBConvBlock(nn.Module):
    """EfficientNet 风格的教学版 MBConv。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 4):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels,
        )
        self.se = SEBlock(hidden_channels, reduction=4)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class PatchEmbedding(nn.Module):
    """把图像切成 patch 并映射成 token。"""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """教学版 Transformer Encoder Block。"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = self.norm1(x)
        attention_output, _ = self.attention(attention_input, attention_input, attention_input)
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x


class LayerNorm2d(nn.Module):
    """ConvNeXt 会用到的二维 LayerNorm。"""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """教学版 ConvNeXt block。"""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pointwise1 = nn.Conv2d(channels, 4 * channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise2 = nn.Conv2d(4 * channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        return x + residual
