# Triton 级实现: aten::gelu 的自研设备码路径（thead-ppu 主自研路径）。
#
# 硬约束（本库实测, templates/operator/README「三条硬约束」）:
#   - 启动必须包 device 上下文（known-issues #11），否则首次之后静默 no-op。
#     本栈（thead-ppu dev 容器）无 flag_gems，上下文用 torch.cuda.device；
#     flag_gems 在册的栈优先用其 torch_device_fn（行为等价）。
#   - 纯 elementwise，无 tl.sum 归约，#15a 尾块归约坑不适用；
#     masked load/store 本身安全，仍覆盖非 BLOCK 整倍数维度。
#   - 不用 @triton.autotune（#15b）。
#
# FlagFix/geluFIX（#5867）迁移约定（issue #55 交付要求 2）:
#   根因是 Triton int 字面量指数 × CANN libdevice pow 查表缺 (fp32,int32)
#   条目（边界: ascend 后端）。PPU 栈走 flagtree/triton 3.6.0 PPUDriver，
#   不消费 CANN libdevice，该 KeyError 不可复现——但可迁移的是写法规约:
#   kernel 内超越函数一律传 float 型指数/显式 fp32 中间量，本文件全部
#   用 `x * x` 显式乘法与 fp32 常量，不出现 int 指数调用。
#
# 适用范围（边界见 PLATFORM.md）:
#   - dtype: fp32 / fp16 / bf16（其余走原生回退）
#   - layout: contiguous（非连续走原生回退，不静默拷贝改变语义）
#   - 0 维 / 空张量: 直接回退原生（无元素可写）
from __future__ import annotations

import torch
import triton
import triton.language as tl

_FALLBACK_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
_SUPPORTED = (torch.float32, torch.float16, torch.bfloat16)
_INT32_MAX = 2**31 - 1


def _device_ctx(dev):
    """#11 设备上下文: flag_gems 在册栈用 torch_device_fn，否则 CUDA 兼容层。"""
    try:
        from flag_gems.runtime import torch_device_fn  # noqa: PLC0415
        return torch_device_fn.device(dev)
    except Exception:
        return torch.cuda.device(dev)


def _fallback(self: torch.Tensor, approximate: str) -> torch.Tensor:
    """原生 ATen gelu（后端键 CUDA 之下，不经本注册）。

    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）: 用后端键
    CUDA 直下厂商 kernel——CompositeExplicitAutograd 落到 ATen 组合
    分解, 半精度下与厂商 kernel 数值分化（bf16 LN 差 0.031 同源实证）。
    """
    return torch.ops.aten.gelu.default.redispatch(
        _FALLBACK_KEYSET, self, approximate=approximate
    )


@triton.jit
def _gelu_fwd_none(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    # erf 精确形式; fp32 常量全程显式（geluFIX #5867 写法规约）
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(y_ptr + i, y.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _tanh_stable(x):
    """手写稳定 tanh（known-issues #13 同款口径）。

    PPU triton 3.6.0 的 tl.math 无 tanh（真机编译实测, 2026-11-25;
    erf 存在）。1 - 2/(exp(2|x|)+1) 在 IEEE inf 算术下溢出安全:
    x→+∞ 时 exp→inf、2/inf→0、结果→1; x=±inf/nan 行为与数学一致。
    """
    ax = tl.math.abs(x) if hasattr(tl.math, "abs") else tl.where(x >= 0, x, -x)
    e = tl.math.exp(2.0 * ax)
    t = 1.0 - 2.0 / (e + 1.0)
    return tl.where(x >= 0, t, -t)


@triton.jit
def _gelu_fwd_tanh(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + _tanh_stable(inner))
    tl.store(y_ptr + i, y.to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gelu_bwd_none(g_ptr, x_ptr, dx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    cdf = 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    pdf = tl.math.exp(-0.5 * x * x) * 0.3989422804014327
    d = cdf + x * pdf
    tl.store(dx_ptr + i, (g * d).to(dx_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gelu_bwd_tanh(g_ptr, x_ptr, dx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    g = tl.load(g_ptr + i, mask=m, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    t = _tanh_stable(inner)
    sech2 = 1.0 - t * t
    du = 0.7978845608028654 * (1.0 + 0.134145 * x * x)
    d = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du
    tl.store(dx_ptr + i, (g * d).to(dx_ptr.dtype.element_ty), mask=m)


def _launch(kernel, n, *args):
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](*args, n, BLOCK=BLOCK)


def gelu_triton(self: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Triton-backed ``aten::gelu``（PPU 自研路径, 否则原生回退）。"""
    if not isinstance(self, torch.Tensor):
        raise TypeError("gelu() expects a Tensor as self")
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")

    if (
        self.layout != torch.strided
        or self.dtype not in _SUPPORTED
        or not self.is_contiguous()
        or self.numel() > _INT32_MAX
        or self.numel() == 0
    ):
        return _fallback(self, approximate)

    out = torch.empty_like(self)
    n = self.numel()
    fwd = _gelu_fwd_none if approximate == "none" else _gelu_fwd_tanh
    with _device_ctx(self.device):  # ← 硬约束 #11
        _launch(fwd, n, self, out)
    return out


def gelu_backward_triton(
    grad_output: torch.Tensor, self: torch.Tensor, approximate: str = "none"
) -> torch.Tensor:
    """Triton-backed ``aten::gelu_backward``（与 forward 同边界）。"""
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    if (
        self.layout != torch.strided
        or self.dtype not in _SUPPORTED
        or not self.is_contiguous()
        or not grad_output.is_contiguous()
        or self.dtype != grad_output.dtype
        or self.shape != grad_output.shape
        or self.numel() > _INT32_MAX
        or self.numel() == 0
    ):
        return torch.ops.aten.gelu_backward.default.redispatch(
            _FALLBACK_KEYSET, grad_output, self, approximate=approximate
        )

    dx = torch.empty_like(self)
    n = self.numel()
    bwd = _gelu_bwd_none if approximate == "none" else _gelu_bwd_tanh
    with _device_ctx(self.device):  # ← 硬约束 #11
        _launch(bwd, n, grad_output, self, dx)
    return dx
