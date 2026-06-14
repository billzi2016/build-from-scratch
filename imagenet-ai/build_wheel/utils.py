"""
这个文件放项目里所有模型都会用到的通用工具函数。

这里不放具体模型结构，只放和教学流程、动态加载、随机种子、
参数统计相关的基础能力，避免在 14 个模型文件里重复写样板代码。
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
from pathlib import Path
from types import ModuleType

import torch


def set_seed(seed: int) -> None:
    """统一设置随机种子，保证教学示例可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(text: str) -> int:
    """把字符串稳定地映射成整数，用来生成确定性的伪数据。"""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_parameter_count(parameter_count: int) -> str:
    """把参数量格式化成更易读的文本。"""
    if parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.2f}M"
    if parameter_count >= 1_000:
        return f"{parameter_count / 1_000:.2f}K"
    return str(parameter_count)


def load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """从任意文件路径动态加载模块，解决年份开头文件名无法直接 import 的问题。"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从路径加载模块: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
