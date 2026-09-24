# torch 级实现: 原生 torch.result_type（纯元数据）。
from __future__ import annotations

import torch


def result_type_torch(a, b):
    return torch.result_type(a, b)
