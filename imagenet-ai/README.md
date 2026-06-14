# imagenet-ai

## 项目简介

`imagenet-ai` 是一个面向初学者的教学型项目。

这个项目的目标，不是追求真实 ImageNet 全量训练，也不是追求论文级严格复现，而是希望沿着时间线，把 ImageNet 时代最有代表性的视觉模型一代一代讲清楚。

这里默认直接使用 `torch` 作为底座。

也就是说，这个项目里的 `from scratch` 不是自己重写张量系统，而是：

1. 自己把每个经典模型一层一层搭出来。
2. 自己把前向传播流程写清楚。
3. 不直接调用现成 `AlexNet`、`ResNet`、`ViT`、`EfficientNet` 整模型成品。

这里更关心的是：

1. 每个模型为什么会出现。
2. 它解决了上一代模型的什么问题。
3. 它的核心结构是什么。
4. 初学者怎么顺着代码真正看懂它。

---

## 项目目标

本项目希望完成一条清晰的经典视觉模型演进主线：

1. 从 `AlexNet` 开始理解深度卷积网络为什么在 ImageNet 上出名。
2. 理解 `ZFNet`、`VGG`、`GoogLeNet` 如何逐步推动卷积网络设计。
3. 理解 `ResNet`、`DenseNet`、`ResNeXt`、`SENet` 这类经典结构为什么重要。
4. 理解 `MobileNet`、`ShuffleNet`、`EfficientNet` 这类轻量化或高效化模型的设计思路。
5. 理解 `Vision Transformer`、`DeiT`、`ConvNeXt` 在后期视觉主干演进中的位置。

---

## 已规划的代表模型

当前项目计划覆盖以下代表模型：

1. `2012_alexnet.py`
2. `2013_zfnet.py`
3. `2014_vgg.py`
4. `2014_googlenet.py`
5. `2015_resnet.py`
6. `2016_densenet.py`
7. `2017_resnext.py`
8. `2017_senet.py`
9. `2017_mobilenet.py`
10. `2018_shufflenet.py`
11. `2019_efficientnet.py`
12. `2020_vision_transformer.py`
13. `2021_deit.py`
14. `2022_convnext.py`

这些模型不是为了“堆列表”，而是为了构成一条适合教学的主线。

---

## 目录结构

当前目录结构如下：

```text
imagenet-ai/
├── README.md
├── imagenet_ai_prd.md
├── build_wheel/
│   ├── main.py
│   ├── 2012_alexnet.py
│   ├── 2013_zfnet.py
│   ├── 2014_vgg.py
│   ├── 2014_googlenet.py
│   ├── 2015_resnet.py
│   ├── 2016_densenet.py
│   ├── 2017_resnext.py
│   ├── 2017_senet.py
│   ├── 2017_mobilenet.py
│   ├── 2018_shufflenet.py
│   ├── 2019_efficientnet.py
│   ├── 2020_vision_transformer.py
│   ├── 2021_deit.py
│   └── 2022_convnext.py
└── imagenet/
    ├── train/
    │   └── train_labels.json
    └── test/
        └── test_labels.json
```

各部分职责如下：

1. `README.md`：给人快速看懂这个项目在做什么。
2. `imagenet_ai_prd.md`：约束这个项目后续怎么实现。
3. `build_wheel/`：放真正的 scratch 实现。
4. `main.py`：负责把整个教学流程串起来。
5. `年份_模型名.py`：分别实现每个代表模型。
6. `dataset.py`、`trainer.py`、`metrics.py`：负责统一数据、训练和评价流程。
7. `imagenet/`：放教学版简化数据。

---

## 数据说明

这个项目默认假设有一个教学版的 `imagenet/` 目录：

```text
imagenet/
├── train/
└── test/
```

这里的数据不是为了模拟真实 ImageNet 的完整规模，而是为了让教学流程跑通。

当前采用的是简化思路：

1. `train/` 和 `test/` 中可以只放少量样例图片。
2. 用 `train_labels.json` 和 `test_labels.json` 记录文件名与类别信息。
3. 不追求真实官方格式。
4. 不把时间浪费在无教学价值的数据工程细节上。
5. 当前代码即使没有真实图片，也会根据文件名和类别生成稳定的教学版伪图像张量。

例如：

```json
[
  {"file_name": "image_0001.jpg", "class_id": 0, "class_name": "cat"},
  {"file_name": "image_0002.jpg", "class_id": 1, "class_name": "dog"}
]
```

---

## 实现原则

这个项目默认遵守根目录 [root_prd.md](/Users/bizi/Desktop/GitHub/build-from-scratch/root_prd.md) 和本目录 [imagenet_ai_prd.md](/Users/bizi/Desktop/GitHub/build-from-scratch/imagenet-ai/imagenet_ai_prd.md)。

核心原则如下：

1. 以教学为第一目标。
2. 让小白能看懂，比代码写得花更重要。
3. 不要为了炫技牺牲可读性。
4. 默认使用 `torch`，但不直接调用现成同名整模型。
5. 先把结构讲清楚，再考虑补复杂功能。
6. 每个模型都应支持最小 train/test 流程。
7. 训练时默认使用 early stopping。
8. 测试时默认输出 `ACC`、`F1`、`AUC`。

---

## 依赖边界

本项目强调“尽量从零理解模型”，所以默认：

1. 可以使用 Python 标准库。
2. 可以使用基础文件和 JSON 处理能力。
3. 可以直接使用 `torch`、`torch.nn`、`torch.optim`、`torch.utils.data`。
4. 但不能直接调用现成的 `AlexNet`、`ResNet`、`ViT`、`EfficientNet` 等模型实现替代学习过程。

换句话说，本项目的重点不是“把库调通”，而是“把结构讲明白”。

---

## 如何使用

当前阶段，这个项目已经提供：

1. 统一训练入口 `build_wheel/main.py`
2. 14 个教学版代表模型
3. 简化数据读取
4. train / test / early stopping 流程
5. `ACC`、`F1`、`AUC` 指标输出

你可以这样使用：

```bash
python build_wheel/main.py --list
python build_wheel/main.py --model alexnet
python build_wheel/main.py --model resnet --epochs 8 --patience 2
python build_wheel/main.py --model all
```

---

## 后续开发建议

比较合理的学习顺序是：

1. 先做 `2012_alexnet.py`
2. 再做 `2014_vgg.py`
3. 再做 `2015_resnet.py`
4. 再逐步扩展到 `DenseNet`、`MobileNet`、`EfficientNet`
5. 最后再做 `Vision Transformer`、`DeiT`、`ConvNeXt`

这样更符合初学者理解路径。
