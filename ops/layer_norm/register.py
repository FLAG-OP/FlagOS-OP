# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# layer_norm 是标准 aten 公开算子，宿主决定交付路径 = A1。
# thead-ppu 算子本地注册 key = **AutogradCUDA**（ops/sdpa/PLATFORM.md §3c
# 先例; known-issues #19.3 inference_mode 零命中为平台边界）。
# issue #55 要求 layer_norm 与 native_layer_norm **同时验证**——
# 注册公开 API 后, 公开调用会消费本实现（不再自然落到 native_layer_norm
# 注册面）；native_layer_norm 的独立拦截由 ops/native_layer_norm 验证。
#
# autograd: 委托原生——LayerNormBackward 原生节点随 redispatch 自动建立。
#
# 调用计数: pid 分片文件（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"layer_norm": 0}
_COUNT_BASE = os.environ.get("A1_LAYER_NORM_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["layer_norm"] = CALL_COUNT.get("layer_norm", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["layer_norm"] = counts.get("layer_norm", 0) + 1
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


class _LayerNormFn(torch.autograd.Function):
    """原生 forward（借道 native_layer_norm 原生三元组）+ 原生 backward。"""

    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        try:
            from .kernel.thead_ppu import layer_norm_stats_ppu
        except ImportError:
            from kernel.thead_ppu import layer_norm_stats_ppu
        out, mean, rstd = layer_norm_stats_ppu(
            input, normalized_shape, weight, bias, eps)
        ctx.normalized_shape = list(normalized_shape)
        ctx.save_for_backward(input, mean, rstd, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        input, mean, rstd, weight, bias = ctx.saved_tensors
        try:
            from .kernel.thead_ppu import layer_norm_backward_ppu
        except ImportError:
            from kernel.thead_ppu import layer_norm_backward_ppu
        need_w = weight is not None and ctx.needs_input_grad[2]
        need_b = bias is not None and ctx.needs_input_grad[3]
        gi, gw, gb = layer_norm_backward_ppu(
            grad_out.contiguous(), input, ctx.normalized_shape,
            mean, rstd, weight, bias, (True, need_w, need_b))
        return (gi, None,
                gw if need_w else None,
                gb if need_b else None,
                None, None)


def layer_norm_aten(input, normalized_shape, weight=None, bias=None,
                    eps=1e-5, cudnn_enable=False):
    """aten::layer_norm(Tensor input, int[] normalized_shape,
    Tensor? weight=None, Tensor? bias=None, float eps=1e-5,
    bool cudnn_enable=False) -> Tensor"""
    _bump()
    if torch.is_grad_enabled() and isinstance(input, torch.Tensor) \
            and (input.requires_grad
                 or (weight is not None and weight.requires_grad)
                 or (bias is not None and bias.requires_grad)):
        return _LayerNormFn.apply(input, normalized_shape, weight, bias, eps)
    try:
        from .kernel.thead_ppu import layer_norm_ppu
    except ImportError:
        from kernel.thead_ppu import layer_norm_ppu
    return layer_norm_ppu(input, normalized_shape, weight, bias, eps)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 layer_norm 到 aten；Library 必须保持引用（thead 范式:
    AutogradCUDA + 自带反向, 理由同 ops/native_layer_norm/register.py）。"""
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r} 需要可用的 PPU 设备（thead-ppu）")
    fn = layer_norm_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return layer_norm_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("layer_norm", fn, dispatch_key)
    return _LIB


register = register_a1
