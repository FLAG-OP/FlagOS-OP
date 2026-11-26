# 注册: 路线 A1（torch 算子替换 · aten 宿主）。
#
# matmul 是标准 aten 公开算子，宿主决定交付路径 = A1。
# thead-ppu 算子本地注册 key = **AutogradCUDA**（ops/sdpa/PLATFORM.md §3c
# 先例; known-issues #19.3 inference_mode 零命中为平台边界）。
#
# ⚠️ 公共目录冲突登记（issue #55 明确排除项条款）: ops/matmul 公共骨架
# 在 main 落地; PR #35（昆仑芯 P800）/ PR #53（寒武纪 MLU590）在途,
# 其平台范围与 perf 注册须在合入时按 MERGE.md 七步流程合并, PPU 范围
# 由本目录 PLATFORM.md 单独登记, 不互相推广。
#
# autograd: 委托原生——原生 autograd 节点随后端键 redispatch 自动建立。
#
# 调用计数: pid 分片文件（docs/testing.md「pid 分片」）。
from __future__ import annotations

import json
import os

import torch

CALL_COUNT = {"matmul": 0}
_COUNT_BASE = os.environ.get("A1_MATMUL_COUNT_FILE")
COUNT_FILE = f"{_COUNT_BASE}.{os.getpid()}" if _COUNT_BASE else None

_LIB = None


def _bump() -> None:
    CALL_COUNT["matmul"] = CALL_COUNT.get("matmul", 0) + 1
    if COUNT_FILE:
        counts = {}
        try:
            with open(COUNT_FILE) as f:
                counts = json.load(f)
        except (OSError, ValueError):
            pass
        counts["matmul"] = counts.get("matmul", 0) + 1
        with open(COUNT_FILE, "w") as f:
            json.dump(counts, f)


_BACKENDS = {"thead-ppu": "thead_ppu"}


def _backend_for(dispatch_key: str | None, platform=None):
    if platform is not None:
        if platform not in _BACKENDS:
            raise RuntimeError(
                f"未知 platform={platform!r}; 已支持: {list(_BACKENDS)}"
                "（新平台接入见 ops/matmul/MERGE.md 与 PR #35/#53 合并流程）")
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


class _MatmulFn(torch.autograd.Function):
    """原生 forward（CUDA redispatch 下探厂商 GEMM）+ 分形态组合反向。

    反向覆盖（aten::matmul 导数公式的主形态; backward 内调用走
    no_grad, 且 aten::mm/matmul 与本注册不同名无递归）:
      - 2D @ 2D:            dA = g @ Bᵀ;       dB = Aᵀ @ g
      - ND(...) @ 2D:       dA = (g @ Bᵀ);     dB = A2ᵀ @ g2（reshape 2D）
      - 2D @ ND(广播批维):  dA = Aᵀ @ g2;      dB = g @ Aᵀ（批广播后逐批）
      - ND @ ND 同批:       dA = g @ Bᵀ;       dB = Aᵀ @ g（逐批）
      - 含广播批维: 先 expand 侧对批维求和还原。
      - 1D 形态 reshape 为 2D 处理后还原。
    """

    @staticmethod
    def forward(ctx, a, b):
        try:
            from .kernel.thead_ppu import matmul_ppu
        except ImportError:
            from kernel.thead_ppu import matmul_ppu
        ctx.save_for_backward(a, b)
        return matmul_ppu(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        with torch.no_grad():
            da, db = _matmul_backward(g, a, b,
                                      ctx.needs_input_grad[0],
                                      ctx.needs_input_grad[1])
        return da, db


def _matmul_backward(g, a, b, need_a, need_b):
    """按 aten::matmul 导数公式分形态组合（与 torch native 语义一致）。"""
    # 1D 形态: 先升 2D
    a1d = a.dim() == 1
    b1d = b.dim() == 1
    a2 = a.reshape(1, -1) if a1d else a
    b2 = b.reshape(-1, 1) if b1d else b
    g2 = g
    if a1d and b1d:          # (K)@(K) -> ()
        g2 = g.reshape(1, 1)
    elif a1d:                # (K)@(K,N) -> (N)
        g2 = g.reshape(1, -1)
    elif b1d:                # (M,K)@(K) -> (M)
        g2 = g.reshape(-1, 1)

    # 广播批维: 确定 a/b 的批形状
    da = db = None
    if a2.dim() >= 2 and b2.dim() >= 2:
        M, K = a2.shape[-2], a2.shape[-1]
        N = b2.shape[-1]
        a_bat = a2.shape[:-2]
        b_bat = b2.shape[:-2]
        g2 = g2.reshape(*g2.shape[:-2], M, N) if g2.dim() > 2 else \
            g2.reshape(*a_bat, M, N) if a_bat else g2
        # 统一批（broadcast 形状）
        g_bat = g2.shape[:-2]
        bat = torch.broadcast_shapes(a_bat, b_bat, tuple(g_bat))
        gE = g2.expand(*bat, M, N)
        aE = a2.expand(*bat, M, K) if a_bat != bat else a2
        bE = b2.expand(*bat, K, N) if b_bat != bat else b2
        gF = gE.reshape(-1, M, N)
        aF = aE.reshape(-1, M, K)
        bF = bE.reshape(-1, K, N)
        if need_a:
            daF = gF @ bF.transpose(-1, -2)          # (B,M,K)
            da = daF.reshape(*bat, M, K)
            # 还原 a 的批（被广播/求和）
            if a_bat != bat:
                sum_dims = _batch_sum_dims(bat, a_bat, a2.dim())
                da = da.sum(dim=sum_dims, keepdim=True).expand_as(a2) \
                    if sum_dims else da.reshape(a2.shape)
            if a1d:
                da = da.reshape(-1)
        if need_b:
            dbF = aF.transpose(-1, -2) @ gF          # (B,K,N)
            db = dbF.reshape(*bat, K, N)
            if b_bat != bat:
                sum_dims = _batch_sum_dims(bat, b_bat, b2.dim())
                db = db.sum(dim=sum_dims, keepdim=True).expand_as(b2) \
                    if sum_dims else db.reshape(b2.shape)
            if b1d:
                db = db.reshape(-1)
    return da, db


def _batch_sum_dims(bat, sub_bat, ndim):
    """求 expand→被广播侧需要 sum 的维索引（从左对齐批维）。"""
    if len(sub_bat) == len(bat):
        return [i for i, (x, y) in enumerate(zip(bat, sub_bat))
                if x != y and y == 1]
    off = len(bat) - len(sub_bat)
    return list(range(off)) + [off + i for i, (x, y)
                               in enumerate(zip(bat[off:], sub_bat))
                               if y == 1 and x != 1]


def matmul_aten(self, other):
    """aten::matmul(Tensor self, Tensor other) -> Tensor"""
    _bump()
    if torch.is_grad_enabled() and isinstance(self, torch.Tensor) \
            and (self.requires_grad or other.requires_grad):
        return _MatmulFn.apply(self, other)
    try:
        from .kernel.thead_ppu import matmul_ppu
    except ImportError:
        from kernel.thead_ppu import matmul_ppu
    return matmul_ppu(self, other)


def register_a1(dispatch_key: str = "AutogradCUDA",
                counter: dict | None = None,
                platform=None) -> torch.library.Library:
    """注册 matmul 到 aten；Library 必须保持引用（thead 范式:
    AutogradCUDA + 分形态自带反向, 理由同 ops/native_layer_norm/register.py）。"""
    global _LIB
    backend = _backend_for(dispatch_key, platform)
    if not _probe_ok(backend):
        raise RuntimeError(
            f"register_a1 backend={backend!r} 需要可用的 PPU 设备（thead-ppu）")
    fn = matmul_aten
    if counter is not None:
        def fn(*args, **kwargs):
            counter["n"] = counter.get("n", 0) + 1
            return matmul_aten(*args, **kwargs)
    _LIB = torch.library.Library("aten", "IMPL")
    _LIB.impl("matmul", fn, dispatch_key)
    return _LIB


register = register_a1
