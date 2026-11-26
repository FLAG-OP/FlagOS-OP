# 注册: 路线 A1（torch 算子替换 · aten 私有稀疏原语宿主）。
#
# 宿主绑定（issue #57 实现范围 1）: torch.sparse.mm 默认入口绑定
# ``aten::_sparse_mm``（torch/sparse/__init__.py:77）; torch.spmm 是
# torch.mm 的别名（→ aten::mm, 保留稠密行为不注册——issue 明确要求）。
# 故本算子注册 **aten::_sparse_mm**。
#
# thead-ppu 注册 key = **AutogradCUDA**（thead 范式 + 自带反向, 理由与
# 递归/autocast 实证同 ops/native_layer_norm/register.py 文件头;
# known-issues #19.3 inference_mode 零命中为平台边界）。
#
# autograd: 自建反向（原生稀疏 autograd 在 PPU 不可用——cusparse 缺口,
# 见 kernel/thead_ppu.py 文件头取证）; d_vals/dB 走 COO 组合
# （ops/spmm/reference.py 同式, 原生 index_add/sum 组合无递归）。
#
# 调用计数: pid 分片文件（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"_sparse_mm": 0}
_COUNT_BASE = os.environ.get("A1_SPARSE_MM_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["_sparse_mm"] = CALL_COUNT.get("_sparse_mm", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["_sparse_mm"] = counts.get("_sparse_mm", 0) + 1
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


class _SparseMmFn(torch.autograd.Function):
    """index_add 组合 forward + COO 组合反向（稀疏结构视为常量）。"""

    @staticmethod
    def forward(ctx, csr, dense):
        try:
            from .kernel.thead_ppu import spmm_ppu
        except ImportError:
            from kernel.thead_ppu import spmm_ppu
        ctx.save_for_backward(csr, dense)
        return spmm_ppu(csr, dense)

    @staticmethod
    def backward(ctx, grad_out):
        csr, dense = ctx.saved_tensors
        try:
            from .kernel.thead_ppu import spmm_backward_ppu
        except ImportError:
            from kernel.thead_ppu import spmm_backward_ppu
        d_vals, dB = spmm_backward_ppu(csr, dense, grad_out.contiguous())
        # dA: CSR values 张量（稀疏结构常量, autograd 语义）
        dA = torch.sparse_csr_tensor(
            csr.crow_indices(), csr.col_indices(), d_vals, csr.shape,
            device=csr.device) if ctx.needs_input_grad[0] else None
        return dA, (dB if ctx.needs_input_grad[1] else None)


def _sparse_mm_aten(self, dense):
    """aten::_sparse_mm(Tensor self, Tensor dense) -> Tensor"""
    _bump()
    if torch.is_grad_enabled() and isinstance(self, torch.Tensor) \
            and (dense.requires_grad or self.values().requires_grad):
        return _SparseMmFn.apply(self, dense)
    try:
        from .kernel.thead_ppu import spmm_ppu
    except ImportError:
        from kernel.thead_ppu import spmm_ppu
    return spmm_ppu(self, dense)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 _sparse_mm 到 aten；Library 必须保持引用。"""
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r} 需要可用的 PPU 设备（thead-ppu）")
    fn = _sparse_mm_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return _sparse_mm_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("_sparse_mm", fn, dispatch_key)
    return _LIB


# 统一入口别名（与 A2/B 插件约定对齐）
register = register_a1
