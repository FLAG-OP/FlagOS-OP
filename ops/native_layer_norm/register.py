# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# native_layer_norm 是标准 aten 私有原语（公开 layer_norm 的底层），
# 宿主决定交付路径 = A1。thead-ppu 注册 key = **AutogradCUDA**
# （ops/sdpa/PLATFORM.md §3c thead 范式; known-issues #19.3
# inference_mode 零命中为平台边界, 如实登记）。
#
# ⚠️ 为什么不用基键 CUDA（真机实证 2026-11-25, probes 同源取证）:
#   1) CUDA 键注册会**覆盖**该栈挂在 CUDA 键上的 autocast 粘合
#      （autocast_mode.cpp:169 同键, 覆盖告警实证）→ autograd 断链;
#   2) CUDA 键注册后, CUDA 键 redispatch 重入自身（RecursionError）,
#      CompositeExplicitAutograd 落到组合分解、半精度数值分化
#      （bf16 LN 差 0.031）→ 委托只能下探 AutogradCUDA 之下的厂商
#      kernel。故按仓库 thead 范式: AutogradCUDA 注册 + 自带反向
#      （native_layer_norm_backward 原生委托, 真机可用）。
#
# issue #55 要求 layer_norm 与 native_layer_norm **同时验证**——
# 两目录的注册/测试一一对应（配对关系见各自 README）。
#
# 调用计数: pid 分片文件（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"native_layer_norm": 0}
_COUNT_BASE = os.environ.get("A1_NATIVE_LAYER_NORM_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["native_layer_norm"] = CALL_COUNT.get("native_layer_norm", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["native_layer_norm"] = counts.get("native_layer_norm", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


_BACKENDS = {"thead-ppu": "thead_ppu"}


def _backend_for(dispatch_key: str | None, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}")
        return _BACKENDS[platform]
    if dispatch_key in ("AutogradCUDA", "CUDA"):
        return "thead_ppu"
    raise RuntimeError(
        f"未知 dispatch_key={dispatch_key!r}; 已支持: AutogradCUDA/CUDA"
        "（或显式传 platform）")


def _probe_ok(backend: str) -> bool:
    if backend != "thead_ppu":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return any(
            "PPU" in torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


class _NativeLayerNormFn(torch.autograd.Function):
    """原生 forward（CUDA redispatch）+ 原生 backward 委托。

    mean/rstd 为 fp32 统计量（真机平台口径: PPU 原生 mean/rstd 恒 fp32,
    keepdim 形状——backward 原语按原样消费）。
    """

    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        try:
            from .kernel.thead_ppu import native_layer_norm_ppu
        except ImportError:
            from kernel.thead_ppu import native_layer_norm_ppu
        out, mean, rstd = native_layer_norm_ppu(
            input, normalized_shape, weight, bias, eps)
        ctx.normalized_shape = list(normalized_shape)
        ctx.eps = eps
        ctx.save_for_backward(input, mean, rstd, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input, mean, rstd, weight, bias = ctx.saved_tensors
        try:
            from .kernel.thead_ppu import native_layer_norm_backward_ppu
        except ImportError:
            from kernel.thead_ppu import native_layer_norm_backward_ppu
        need_w = weight is not None and ctx.needs_input_grad[2]
        need_b = bias is not None and ctx.needs_input_grad[3]
        if not (need_w or need_b):
            weight = bias = None
        gi, gw, gb = native_layer_norm_backward_ppu(
            grad_out.contiguous(), input, ctx.normalized_shape,
            mean, rstd, weight, bias, (True, need_w, need_b))
        return (gi, None,
                gw if need_w else None,
                gb if need_b else None,
                None)


def native_layer_norm_aten(input, normalized_shape, weight, bias, eps):
    """aten::native_layer_norm(Tensor input, SymInt[] normalized_shape,
    Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"""
    _bump()
    if torch.is_grad_enabled() and isinstance(input, torch.Tensor) \
            and (input.requires_grad
                 or (weight is not None and weight.requires_grad)
                 or (bias is not None and bias.requires_grad)):
        # 三返回值语义: mean/rstd 也需可微暴露——Function 只回 out,
        # mean/rstd 以无梯度张量返回（与原生 autograd 语义一致:
        # mean/rstd 无梯度边）。
        out = _NativeLayerNormFn.apply(input, normalized_shape, weight,
                                       bias, eps)
        with torch.no_grad():
            _, mean, rstd = _forward_stats(input, normalized_shape, eps)
        return out, mean, rstd
    try:
        from .kernel.thead_ppu import native_layer_norm_ppu
    except ImportError:
        from kernel.thead_ppu import native_layer_norm_ppu
    return native_layer_norm_ppu(input, normalized_shape, weight, bias, eps)


def _forward_stats(input, normalized_shape, eps):
    """no-grad 统计量重算（mean/rstd 返回语义; 复用 backend 委托）。"""
    with torch.no_grad():
        try:
            from .kernel.thead_ppu import native_layer_norm_ppu
        except ImportError:
            from kernel.thead_ppu import native_layer_norm_ppu
        return native_layer_norm_ppu(input, normalized_shape, None, None, eps)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 native_layer_norm 到 aten；Library 必须保持引用。

    dispatch_key: 默认 AutogradCUDA（thead 范式, 自带反向;
    见文件头「为什么不用基键 CUDA」）。"""
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r} 需要可用的 PPU 设备（thead-ppu）")
    fn = native_layer_norm_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return native_layer_norm_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("native_layer_norm", fn, dispatch_key)
    return _LIB


register = register_a1
