"""Ascend910 embedding backend.

wt <wangt635@ustc.edu.cn>

架构对齐 p800_kunlunxin backend（PR#3 确立的交付模式）:
生产实现 = 委托 CANN 原生行采集（aten::index_select——本机实测
vocab=131k/dim=4096/tokens=16k fp16 0.185ms，与 F.embedding 同源
同速，数值精确 err=0）；Triton gather 仅保留为可复现平台探针
（P800 实测 Triton gather 慢数十倍，见 ops/embedding/reports/
performance.md，Ascend 预期同款，探针数据见 reports/）。

dense backward: aten::embedding_dense_backward 委托 + 本地参考
（scale_grad_by_freq 模式按逐出现逆频率缩放，语义与
reference.embedding_backward_reference 对齐）。

平台绑定: 仅 npu（torch_npu）设备 + AutogradPrivateUse1 注册点
（sdpa 开发报告 §3.1 的 dispatch 证据链同样适用于 embedding）。
"""
from __future__ import annotations

import importlib.util

import torch

PLATFORM = "ascend910"
SUPPORTED_DEVICE_TYPES = ("npu",)
_HAS_ASCEND_STACK = None


def _has_ascend_stack() -> bool:
    global _HAS_ASCEND_STACK
    if _HAS_ASCEND_STACK is None:
        if importlib.util.find_spec("torch_npu") is None:
            _HAS_ASCEND_STACK = False
        else:
            try:
                import torch_npu  # noqa: F401
                _HAS_ASCEND_STACK = True
            except Exception:
                _HAS_ASCEND_STACK = False
    return _HAS_ASCEND_STACK


def _validate(weight, indices, padding_idx, num_weights):
    if weight.device.type not in SUPPORTED_DEVICE_TYPES \
            or not _has_ascend_stack():
        raise RuntimeError(
            f"embedding 是 PLATFORM={PLATFORM!r} 绑定实现，收到 "
            f"device={weight.device!r} 或缺少 torch_npu。"
            "跨平台选择见 ops/embedding 各 backend / MERGE 指南。")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported floating dtype: {weight.dtype}")
    if indices.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"unsupported indices dtype: {indices.dtype}")
    if weight.device != indices.device:
        raise ValueError("weight and indices must be on the same device")
    if padding_idx is not None and padding_idx != -1:
        if padding_idx < 0 or padding_idx >= num_weights:
            raise IndexError(
                "padding_idx must be -1/None or inside [0, num_weights)")


def embedding(weight: torch.Tensor, indices: torch.Tensor,
              padding_idx: int | None = -1,
              scale_grad_by_freq: bool = False,
              sparse: bool = False) -> torch.Tensor:
    """CANN 原生行采集实现 aten::embedding 前向（查表不受
    padding_idx/scale_grad/sparse 影响——语义见 reference.py）。"""
    # 热路径: 1D token id + 无 padding 语义变换 → 单次原生行采集
    if (weight.dim() == 2 and indices.dim() == 1
            and padding_idx in (-1, None)):
        return torch.ops.aten.index_select(weight, 0, indices)
    _validate(weight, indices, padding_idx, weight.shape[0])
    if weight.dim() != 2:
        raise ValueError("weight must be 2D (num_weights, dim)")
    if indices.numel() and (indices.min().item() < 0
                            or indices.max().item() >= weight.shape[0]):
        raise IndexError("embedding indices are out of range")
    flat = indices.reshape(-1)
    out = torch.ops.aten.index_select(weight, 0, flat)
    return out.view(*indices.shape, weight.shape[-1])


def embedding_backward(grad_output: torch.Tensor,
                        indices: torch.Tensor, num_weights: int,
                        padding_idx: int | None = -1,
                        scale_grad_by_freq: bool = False,
                        sparse: bool = False) -> torch.Tensor:
    """dense 反向: 委托 aten::embedding_dense_backward; 其不支持
    scale_grad_by_freq 时的逐出现逆频率缩放实现（P800 同款模式）。"""
    _validate(grad_output, indices, padding_idx, num_weights)
    if num_weights < 0:
        raise ValueError("num_weights must be non-negative")
    if sparse:
        raise NotImplementedError(
            "sparse embedding backward is outside this delivery")
    if not scale_grad_by_freq and padding_idx in (-1, None):
        # 纯 dense: CANN 原生反向
        # aten::embedding_dense_backward(grad_output, indices,
        # num_weights: SymInt, padding_idx: SymInt,
        # scale_grad_by_freq: bool) -> Tensor  —— 5 参, 无 sparse
        return torch.ops.aten.embedding_dense_backward(
            grad_output, indices, num_weights,
            padding_idx if padding_idx is not None else -1, False)

    # scale_grad_by_freq / padding_idx 路径: 本地聚合实现
    flat_g = grad_output.reshape(-1, grad_output.shape[-1])
    flat_i = indices.reshape(-1)
    dim = flat_g.shape[-1]
    grad = torch.zeros((num_weights, dim), dtype=grad_output.dtype,
                       device=grad_output.device)
    if scale_grad_by_freq:
        # 每个被引用行的出现次数 → 逆频率缩放
        counts = torch.bincount(flat_i, minlength=num_weights)
        inv = torch.where(counts > 0, 1.0 / counts.clamp(min=1),
                          torch.zeros_like(counts, dtype=flat_g.dtype))
        scaled = flat_g * inv[flat_i].unsqueeze(-1).to(flat_g.dtype)
        grad.index_add_(0, flat_i, scaled)
    else:
        grad.index_add_(0, flat_i, flat_g)
    if padding_idx is not None and padding_idx != -1:
        grad[padding_idx].zero_()
    return grad
