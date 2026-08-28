# torch 级 softmax: ATen 原生实现（不产生新设备码，作为对照实现）。
import torch
import torch.nn.functional as F


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """F.softmax 直通（ATen CUDA/XPU kernel）。"""
    return F.softmax(x, dim=-1)
