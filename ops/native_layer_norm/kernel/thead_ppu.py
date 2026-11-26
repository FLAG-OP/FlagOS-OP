"""T-head PPU (ZW810E) native_layer_norm backend — native 委托（复用路线）。

台账结论（issue #55; probes/thead_dense_p0_probe.py 真机取证, 卡6）:

- ``aten::native_layer_norm`` 原生可用: fp16 实测 err≈9.5e-7（3 返回值
  out/mean/rstd 齐备, 2026-11-25 卡6）。
- LayerNorm 是「归一化统计 + affine」复合——自研 Triton 需 row-wise
  归约（tl.sum, #15a 尾块归约坑）且收益存疑（厂商库强项, 与 sdpa/
  embedding 复用结论同构）; 生产路径**委托原生**，torch_level 提供
  组合对照。
"""
from __future__ import annotations

import torch

PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_PPU_STACK = None


def _has_ppu_stack() -> bool:
    global _HAS_PPU_STACK
    if _HAS_PPU_STACK is None:
        try:
            from triton.runtime import driver
            if "PPU" in type(driver.active).__name__:
                _HAS_PPU_STACK = True
                return True
        except Exception:
            pass
        try:
            _HAS_PPU_STACK = torch.cuda.is_available() and any(
                "PPU" in torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            )
        except Exception:
            _HAS_PPU_STACK = False
    return _HAS_PPU_STACK


def _device_is_ppu(device: torch.device) -> bool:
    if device.type not in SUPPORTED_DEVICE_TYPES:
        return False
    try:
        return "PPU" in torch.cuda.get_device_name(device)
    except Exception:
        return False


def _validate(input: torch.Tensor, weight, bias) -> None:
    if not _has_ppu_stack():
        raise RuntimeError(
            f"native_layer_norm 是 PLATFORM={PLATFORM!r} 绑定实现，"
            "当前环境无 PPU 栈（见 ops/native_layer_norm/PLATFORM.md）")
    ts = [("input", input)] + ([("weight", weight)] if weight is not None else []) \
        + ([("bias", bias)] if bias is not None else [])
    for name, t in ts:
        if t.device.type not in SUPPORTED_DEVICE_TYPES \
                or not _device_is_ppu(t.device):
            raise RuntimeError(
                f"native_layer_norm 是 PLATFORM={PLATFORM!r} 绑定实现，"
                f"{name} 在 {t.device!r}（非 PPU 栈）")
        if t.dtype not in SUPPORTED_DTYPES:
            raise RuntimeError(
                f"native_layer_norm thead backend 不支持 {name}.dtype="
                f"{t.dtype}（其他 dtype 由 register 层走原生回退）")


def native_layer_norm_ppu(input: torch.Tensor, normalized_shape, weight,
                          bias, eps: float = 1e-5):
    """PPU 生产实现: 委托原生 ``aten::native_layer_norm``（后端键 CUDA
    redispatch——厂商 kernel 完好（本算子注册在 AutogradCUDA, 不占 CUDA
    键）, 与直调逐位一致; 返回 (out, mean, rstd) 三元组）。

    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）:
      - 后端键 CUDA: 与直调逐位一致（厂商 kernel 完好时）;
      - CompositeExplicitAutograd: 落到 ATen 组合分解, 半精度下与厂商
        kernel 数值分化（bf16 LN 差 0.031 实证）——**不可用**;
      - 同键（CUDA）注册后 CUDA 键 redispatch 会重入自身（RecursionError
        实证）——故本算子注册 AutogradCUDA（仓库 thead 范式）。
    """
    _validate(input, weight, bias)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.native_layer_norm.default.redispatch(
        keyset, input, list(normalized_shape), weight, bias, eps)


def native_layer_norm_backward_ppu(grad_out: torch.Tensor, input: torch.Tensor,
                                   normalized_shape, mean: torch.Tensor,
                                   rstd: torch.Tensor, weight, bias,
                                   masks=(True, True, True)):
    """PPU 反向: 委托原生 ``aten::native_layer_norm_backward``（真机可用,
    2026-11-25 卡6 实测; 返回 (grad_input, grad_weight, grad_bias)）。"""
    _validate(input, weight, bias)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.native_layer_norm_backward.default.redispatch(
        keyset, grad_out, input, list(normalized_shape), mean, rstd,
        weight, bias, list(masks))
