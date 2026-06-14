"""
这个文件维护 imagenet-ai 项目里所有代表模型的注册表。

主入口不直接写死大量 if/else，而是统一从这里拿模型元数据，
这样教学流程会更清晰，也方便后续继续扩展新的里程碑模型。
"""

from __future__ import annotations

from pathlib import Path


BUILD_WHEEL_DIR = Path(__file__).resolve().parent


MODEL_REGISTRY = [
    {
        "key": "alexnet",
        "year": 2012,
        "display_name": "AlexNet",
        "file_name": "2012_alexnet.py",
        "summary": "经典深层卷积分类网络。",
    },
    {
        "key": "zfnet",
        "year": 2013,
        "display_name": "ZFNet",
        "file_name": "2013_zfnet.py",
        "summary": "AlexNet 的结构调优版本。",
    },
    {
        "key": "vgg",
        "year": 2014,
        "display_name": "VGG",
        "file_name": "2014_vgg.py",
        "summary": "用小卷积核堆出更深更规整的网络。",
    },
    {
        "key": "googlenet",
        "year": 2014,
        "display_name": "GoogLeNet",
        "file_name": "2014_googlenet.py",
        "summary": "用 Inception 多分支提升表达能力。",
    },
    {
        "key": "resnet",
        "year": 2015,
        "display_name": "ResNet",
        "file_name": "2015_resnet.py",
        "summary": "用残差连接解决深层网络训练困难。",
    },
    {
        "key": "densenet",
        "year": 2016,
        "display_name": "DenseNet",
        "file_name": "2016_densenet.py",
        "summary": "强调特征复用和梯度流动。",
    },
    {
        "key": "resnext",
        "year": 2017,
        "display_name": "ResNeXt",
        "file_name": "2017_resnext.py",
        "summary": "在残差结构里加入 cardinality。",
    },
    {
        "key": "senet",
        "year": 2017,
        "display_name": "SENet",
        "file_name": "2017_senet.py",
        "summary": "通过通道注意力重标定特征。",
    },
    {
        "key": "mobilenet",
        "year": 2017,
        "display_name": "MobileNet",
        "file_name": "2017_mobilenet.py",
        "summary": "用深度可分离卷积做轻量化。",
    },
    {
        "key": "shufflenet",
        "year": 2018,
        "display_name": "ShuffleNet",
        "file_name": "2018_shufflenet.py",
        "summary": "用分组卷积与通道混洗做轻量化。",
    },
    {
        "key": "efficientnet",
        "year": 2019,
        "display_name": "EfficientNet",
        "file_name": "2019_efficientnet.py",
        "summary": "用 compound scaling 平衡深度宽度分辨率。",
    },
    {
        "key": "vision_transformer",
        "year": 2020,
        "display_name": "Vision Transformer",
        "file_name": "2020_vision_transformer.py",
        "summary": "把图像切成 patch 后送入 Transformer。",
    },
    {
        "key": "deit",
        "year": 2021,
        "display_name": "DeiT",
        "file_name": "2021_deit.py",
        "summary": "教学版视觉 Transformer 蒸馏结构。",
    },
    {
        "key": "convnext",
        "year": 2022,
        "display_name": "ConvNeXt",
        "file_name": "2022_convnext.py",
        "summary": "Transformer 时代重新设计的 CNN。",
    },
]


def get_model_keys() -> list[str]:
    """返回可选模型 key 列表。"""
    return [item["key"] for item in MODEL_REGISTRY]


def get_model_spec(model_key: str) -> dict:
    """根据 key 获取单个模型配置。"""
    for item in MODEL_REGISTRY:
        if item["key"] == model_key:
            return item
    raise KeyError(f"未知模型: {model_key}")
