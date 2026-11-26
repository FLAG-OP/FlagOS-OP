# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# softmax 是标准 aten 公开算子，宿主决定交付路径 = A1。
# thead-ppu 算子本地注册 key = **AutogradCUDA**（ops/sdpa/PLATFORM.md §3c
# 先例; known-issues #19.3 inference_mode 零命中为平台边界）。
# issue #55 要求 softmax 与 _softmax 同时验证并记录调用关系——
# 公开 softmax 的注册实现委托原生 ``aten::softmax.int_dtype`` 重载
# （半精度 + dtype 语义经该重载表达）; 原语 _softmax 的独立注册在
# ops/_softmax（二者关系: F.softmax → aten::softmax → _softmax 原语,
# 平台绑定见各自 PLATFORM.md）。
#
# autograd: 委托原生——SoftmaxBackward 原生节点随 redispatch 自动建立。
#
# 调用计数: pid 分片文件（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"softmax": 0}
_COUNT_BASE = os.environ.get("A1_SOFTMAX_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["softmax"] = CALL_COUNT.get("softmax", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["softmax"] = counts.get("softmax", 0) + 1
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


class _SoftmaxFn(torch.autograd.Function):
    """原生 forward（CUDA redispatch 下探厂商 kernel）+ 标准解析反向。

    softmax 反向（数学恒等式, 与 ATen SoftmaxBackward 同式）:
        dX = Y ⊙ (g − (Y⊙g).sum(dim, keepdim))
    其中 Y 为前向输出（减 max 稳定归一化的结果, 保存复用避免重算）。
    dtype 参数（half_to_float 语义）反向: g 先 cast 回输入 dtype 口径
    （ATen 语义: 输出 fp32 时梯度先在 fp32 域计算再落回输入 dtype）。
    """

    @staticmethod
    def forward(ctx, self, dim, dtype):
        try:
            from .kernel.thead_ppu import softmax_ppu
        except ImportError:
            from kernel.thead_ppu import softmax_ppu
        out = softmax_ppu(self, dim, dtype)
        ctx.dim = dim
        ctx.self_dtype = self.dtype
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        (out,) = ctx.saved_tensors
        with torch.no_grad():
            inner = (out * g).sum(dim=ctx.dim, keepdim=True)
            grad_self = out * (g - inner)
            if grad_self.dtype != ctx.self_dtype:
                grad_self = grad_self.to(ctx.self_dtype)
        return grad_self, None, None


def softmax_aten(self, dim, dtype=None):
    """aten::softmax(Tensor self, int dim, *, ScalarType? dtype=None)"""
    _bump()
    if torch.is_grad_enabled() and isinstance(self, torch.Tensor) \
            and self.requires_grad:
        return _SoftmaxFn.apply(self, dim, dtype)
    try:
        from .kernel.thead_ppu import softmax_ppu
    except ImportError:
        from kernel.thead_ppu import softmax_ppu
    return softmax_ppu(self, dim, dtype)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 softmax 到 aten；Library 必须保持引用（thead 范式:
    AutogradCUDA + 解析反向, 理由同 ops/native_layer_norm/register.py）。"""
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r} 需要可用的 PPU 设备（thead-ppu）")
    fn = softmax_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return softmax_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    # ⚠️ aten::softmax 无 default 重载（本栈重载集 int/Dimname/int_out,
    # 真机实测）——必须按重载名绑定 "softmax.int"（F.softmax 的实际路由
    # 落点）; 按 "softmax" 绑定会注册到无 schema 空壳, 调用永不命中
    # （op 层"未被拦截"实证）。
    _LIB.impl("softmax.int", fn, dispatch_key)
    return _LIB


register = register_a1
