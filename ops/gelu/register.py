# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# gelu 是标准 aten 算子，宿主决定交付路径 = A1（本库集成指南）。
# thead-ppu 算子本地注册 key = **AutogradCUDA**（ops/sdpa/PLATFORM.md §3c
# 先例: 探针 6/7 命中, 唯一零命中是 inference_mode——known-issues #19.3
# 通病, 如实登记为平台边界; 中央 YAML 基键 CUDA 有意分层, 见该文件
# §3c-1）。PPU 经 CUDA 兼容层直曝 device.type="cuda"。
#
# autograd: AutogradCUDA 是 autograd 键——注册即**替换** torch 为 gelu
# 内置的 autograd kernel, 故必须用 autograd.Function 自建反向节点
# （sdpa 同款, 见其 register.py）。反向用自研 gelu_backward_triton
# （解析公式, FlagFix #5867 涉及的 fused 反向路径按本算子模板重验）。
#
# 调用计数（内存 + 可选文件落地）: 框架级测试跨进程读取, pid 分片避免
# TP 多 worker 并发写竞争（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"gelu": 0}
_COUNT_BASE = os.environ.get("A1_GELU_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["gelu"] = CALL_COUNT.get("gelu", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["gelu"] = counts.get("gelu", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


# ── 平台选择（thead-ppu 首平台; 第二平台出现时按 embedding 范式扩表）──
_BACKENDS = {"thead-ppu": "thead_ppu"}


def _backend_for(dispatch_key: str | None, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
                "（新平台接入见 ops/sdpa/PLATFORM.md §5 范式）")
        return _BACKENDS[platform]
    # dispatch_key 推断: PPU 经 CUDA 兼容层; AutogradCUDA/CUDA 均视为 PPU 候选,
    # 由 _probe_ok 的设备名校验兜底。
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


class _GeluFn(torch.autograd.Function):
    """自研 forward + 解析 backward（替代 torch 内置 gelu autograd 节点）。"""

    @staticmethod
    def forward(ctx, x, approximate):
        ctx.approximate = approximate
        ctx.save_for_backward(x)
        # 包态优先、脚本态 fallback（多算子同进程, type_as 同款）
        try:
            from .kernel.triton_level import gelu_triton
        except ImportError:
            from kernel.triton_level import gelu_triton
        return gelu_triton(x, approximate)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        try:
            from .kernel.triton_level import gelu_backward_triton
        except ImportError:
            from kernel.triton_level import gelu_backward_triton
        grad_x = gelu_backward_triton(grad_out.contiguous(), x.contiguous(),
                                      ctx.approximate)
        return grad_x, None


def gelu_aten(self, approximate="none"):
    """aten::gelu(Tensor self, *, str approximate='none') -> Tensor"""
    _bump()
    if torch.is_grad_enabled() and isinstance(self, torch.Tensor) \
            and self.requires_grad:
        return _GeluFn.apply(self, approximate)
    try:
        from .kernel.triton_level import gelu_triton
    except ImportError:
        from kernel.triton_level import gelu_triton
    return gelu_triton(self, approximate)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 gelu 到 aten；Library 对象必须保持引用（否则被回收）。

    dispatch_key: 默认 AutogradCUDA（thead-ppu 算子本地键, sdpa §3c 先例）。
    counter: 可选 dict, 命中计数写入 counter["n"]（op 层拦截验证）。
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）。
    """
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r}, dispatch_key={dispatch_key!r} "
            f"需要可用的 PPU 设备（thead-ppu）")
    fn = gelu_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return gelu_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("gelu", fn, dispatch_key)
    return _LIB


# 统一入口别名（与 A2/B 插件约定对齐，便于框架级注入）
register = register_a1
