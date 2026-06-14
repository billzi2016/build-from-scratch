"""
这个文件实现教学版数据集读取逻辑。

用户已经明确说过，不需要纠结真实 ImageNet 的巨大数据格式，
所以这里采用“标签 JSON + 伪图像张量”的方式，重点是让训练、
测试和网络结构演示能跑通，而不是做复杂数据工程。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils import stable_hash


class ToyImageNetDataset(Dataset):
    """
    教学版 ImageNet 数据集。

    这里默认根据文件名和类别生成确定性的伪图像张量。
    这样即使目录里没有真实图片文件，训练和测试流程也能完整跑通。
    """

    def __init__(self, labels_path: Path, image_size: int = 64):
        self.labels_path = Path(labels_path)
        self.image_size = image_size

        with self.labels_path.open("r", encoding="utf-8") as file:
            self.samples = json.load(file)

        if not self.samples:
            raise ValueError(f"标签文件为空: {self.labels_path}")

        self.class_names = self._build_class_names()
        self.num_classes = len(self.class_names)

    def _build_class_names(self) -> list[str]:
        class_name_by_id = {}
        for sample in self.samples:
            class_name_by_id[int(sample["class_id"])] = sample["class_name"]
        return [class_name_by_id[class_id] for class_id in sorted(class_name_by_id)]

    def _generate_tensor_from_sample(self, sample: dict) -> torch.Tensor:
        """
        根据文件名和类别生成稳定伪图像。

        这样做的目的不是模拟真实图像，而是提供“有可学模式”的输入，
        方便不同网络都能跑最小训练与测试流程。
        """
        class_id = int(sample["class_id"])
        file_name = sample["file_name"]
        seed = stable_hash(f"{file_name}-{class_id}")
        generator = torch.Generator().manual_seed(seed)

        image = torch.rand((3, self.image_size, self.image_size), generator=generator) * 0.15

        patch_size = max(6, self.image_size // 5)
        row_start = (class_id * 11) % (self.image_size - patch_size)
        col_start = (class_id * 17) % (self.image_size - patch_size)
        color_channel = class_id % 3

        image[color_channel, row_start : row_start + patch_size, col_start : col_start + patch_size] += 0.65
        image[(color_channel + 1) % 3, :, col_start : col_start + 2] += 0.20
        image[(color_channel + 2) % 3, row_start : row_start + 2, :] += 0.20

        return image.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = self._generate_tensor_from_sample(sample)
        label = torch.tensor(int(sample["class_id"]), dtype=torch.long)
        return image, label


def load_datasets(data_root: Path, image_size: int = 64) -> tuple[ToyImageNetDataset, ToyImageNetDataset]:
    """加载 train / test 两个教学版数据集。"""
    data_root = Path(data_root)
    train_dataset = ToyImageNetDataset(data_root / "train" / "train_labels.json", image_size=image_size)
    test_dataset = ToyImageNetDataset(data_root / "test" / "test_labels.json", image_size=image_size)
    return train_dataset, test_dataset
