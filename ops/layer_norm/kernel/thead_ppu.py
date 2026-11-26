"""T-head PPU (ZW810E) layer_norm backend — native 委托（复用路线）。

台账结论（issue #55; probes/thead_dense_p0_probe.py 真机取证, 卡6）:

- ``aten::layer_norm`` 原生可用: fp16 实测 err=0.0（2026-11-25 卡6）。
- 公开 API 内部消费 native_layer_norm 原语; 生产路径委托原生——
  归一化统计是厂商库强项, 自研 Triton 需 row-wise tl.sum（#15a 尾块
  归约坑）且无收益（与 sdpa/embedding 复用结论同构）。
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
            f"layer_norm 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 "
            "PPU 栈（见 ops/layer_norm/PLATFORM.md）")
    ts = [("input", input)] + ([("weight", weight)] if weight is not None else []) \
        + ([("bias", bias)] if bias is not None else [])
    for name, t in ts:
        if t.device.type not in SUPPORTED_DEVICE_TYPES \
                or not _device_is_ppu(t.device):
            raise RuntimeError(
                f"layer_norm 是 PLATFORM={PLATFORM!r} 绑定实现，"
                f"{name} 在 {t.device!r}（非 PPU 栈）")
        if t.dtype not in SUPPORTED_DTYPES:
            raise RuntimeError(
                f"layer_norm thead backend 不支持 {name}.dtype={t.dtype}"
                "（其他 dtype 由 register 层走原生回退）")


def layer_norm_ppu(input: torch.Tensor, normalized_shape, weight, bias,
                   eps: float = 1e-5) -> torch.Tensor:
    """PPU 生产实现: 委托原生 ``aten::layer_norm``（后端键 CUDA
    redispatch——厂商 kernel 完好, 与直调逐位一致）。

    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）:
      - 后端键 CUDA: 与直调逐位一致;
      - CompositeExplicitAutograd: ATen 组合分解, 半精度数值分化（不可用）;
      - 同键（CUDA）注册后重入自身（RecursionError）——本算子注册
        AutogradCUDA（仓库 thead 范式）。
    """
    _validate(input, weight, bias)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.layer_norm.default.redispatch(
        keyset, input, list(normalized_shape), weight, bias, eps, False)


def layer_norm_stats_ppu(input: torch.Tensor, normalized_shape, weight, bias,
                         eps: float = 1e-5):
    """forward 统计对: 借道 ``aten::native_layer_norm`` 原生三元组
    （layer_norm 的 mean/rstd 即原语返回; 真机原生 err=0.0）。"""
    _validate(input, weight, bias)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.native_layer_norm.default.redispatch(
        keyset, input, list(normalized_shape), weight, bias, eps)


def layer_norm_backward_ppu(grad_out: torch.Tensor, input: torch.Tensor,
                            normalized_shape, mean: torch.Tensor,
                            rstd: torch.Tensor, weight, bias,
                            masks=(True, True, True)):
    """PPU 反向: 委托原生 ``aten::native_layer_norm_backward``（该栈未
    暴露 aten::layer_norm_backward Python 绑定, 真机实测;
    layer_norm = native_layer_norm 包装, 反向同式）。"""
    _validate(input, weight, bias)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.native_layer_norm_backward.default.redispatch(
        keyset, grad_out, input, list(normalized_shape), mean, rstd,
        weight, bias, list(masks))
