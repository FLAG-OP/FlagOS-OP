# 注册: 路线 A1（torch 算子替换 · aten 私有原语宿主）。
#
# _softmax 是 softmax 的私有原语（公开 API 的内部落点）。
# thead-ppu 注册 key = **基键 CUDA**（与 ops/softmax 同分层口径: 委托型
# 注册在 autograd 节点之下, 原生 backward 保留、inference_mode 拦截;
# scripts/smoke_dispatch_key.py 三口径 7/7 实测）。
#
# 与公开 softmax 的注册边界（issue #55 要求记录）:
#   - ops/softmax 注册 aten::softmax（公开 API 面）;
#   - 本目录注册 aten::_softmax（原语面）——两注册互不遮蔽（不同算子名）;
#   - 公开调用链: F.softmax → aten::softmax（ops/softmax 实现）→ 内部
#     原生落点 aten::_softmax（本实现, 委托原生）。钉选/计数按两个注册
#     分别归因, 不混算。
#
# autograd: 委托原生——SoftmaxBackward 原生节点随基键注册自动保留。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"_softmax": 0}
_COUNT_BASE = os.environ.get("A1__SOFTMAX_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["_softmax"] = CALL_COUNT.get("_softmax", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["_softmax"] = counts.get("_softmax", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


_BACKENDS = {"thead-ppu": "thead_ppu"}


def _backend_for(dispatch_key: str | None, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}")
        return _BACKENDS[platform]
    if dispatch_key in ("CUDA", "AutogradCUDA"):
        return "thead_ppu"
    raise RuntimeError(
        f"未知 dispatch_key={dispatch_key!r}; 已支持: CUDA/AutogradCUDA"
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


class _SoftmaxPrimFn(torch.autograd.Function):
    """原生 forward（CUDA redispatch）+ 标准解析反向（输出保存复用）。

    dX = Y ⊙ (g − (Y⊙g).sum(dim, keepdim)); h2f=True 时输出 fp32,
    梯度在 fp32 域计算后落回输入 dtype（ATen 语义）。"""

    @staticmethod
    def forward(ctx, self, dim, half_to_float):
        try:
            from .kernel.thead_ppu import _softmax_ppu
        except ImportError:
            from kernel.thead_ppu import _softmax_ppu
        out = _softmax_ppu(self, dim, half_to_float)
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


def _softmax_aten(self, dim, half_to_float=False):
    """aten::_softmax(Tensor self, int dim, bool half_to_float=False)"""
    _bump()
    if torch.is_grad_enabled() and isinstance(self, torch.Tensor) \
            and self.requires_grad:
        return _SoftmaxPrimFn.apply(self, dim, half_to_float)
    try:
        from .kernel.thead_ppu import _softmax_ppu
    except ImportError:
        from kernel.thead_ppu import _softmax_ppu
    return _softmax_ppu(self, dim, half_to_float)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 _softmax 到 aten；Library 必须保持引用。"""
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r} 需要可用的 PPU 设备（thead-ppu）")
    fn = _softmax_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return _softmax_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("_softmax", fn, dispatch_key)
    return _LIB


register = register_a1
