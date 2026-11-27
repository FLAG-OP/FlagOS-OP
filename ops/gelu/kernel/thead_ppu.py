"""T-head PPU (ZW810E) gelu 原生对照 backend（native 对照 + 回退底座）。

角色（与 register.py 的 triton 主自研路径配套）:

- 生产主路径是 ``kernel/triton_level.py``（自研 Triton fwd+bwd，
  AutogradCUDA 注册携带解析反向——register.py `_GeluFn`）。本文件
  **不接生产注册**，职责是:
  1. tests 的三方对照「自研 Triton vs 原生 ATen vs fp32 参考」中的
     原生侧（issue #55 交付要求 3: 自研/原生/FlagGems 对照）;
  2. triton 边界外（非连续/奇异 dtype/空张量等）的原生回退底座——
     与 triton_level._fallback 的 redispatch 同 keyset 口径。
- 台账依据（issue #55; probes/thead_dense_p0_probe.py 真机取证, 卡6）:
  ``aten::gelu``（none/tanh）与 ``aten::gelu_backward`` 在 PPU
  Torch-FL 栈原生存在。gelu 是逐 elementwise 算子，无 tl.dot /
  tl.sum，不踩 PPU Triton 的 `tl.dot` 精度降级（sdpa PLATFORM §3c:
  64³ rel≈7e-4）与 #15a 尾块归约坑，故自研 Triton 在本算子有确定性
  与可携带反向的收益；原生路径保留为对照与回退。

FlagFix/geluFIX（gelu-5867，commit f07f222）修复的是 **ascend 后端**
CANN pow 类型问题（FlagGems geglu.py / gelu_and_mul.py / gelu.py），
平台边界是昇腾；对 thead 不作未经核实的直接声明——迁移记录与适用
范围判定见 README.md「FlagFix 迁移记录」节与本算子 PLATFORM.md。
"""
from __future__ import annotations

import torch

PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # PPU torch 经 CUDA 兼容层直曝 cuda:N
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_APPROXIMATE = ("none", "tanh")
_HAS_PPU_STACK = None


def _has_ppu_stack() -> bool:
    """进程级 PPU 栈识别（与 embedding/sdpa thead backend 同款双路探测）。"""
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


def _validate(self: torch.Tensor, approximate: str) -> None:
    if not _has_ppu_stack():
        raise RuntimeError(
            f"gelu 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 PPU 栈"
            "（误配请检查 profile，见 ops/gelu/PLATFORM.md）")
    if self.device.type not in SUPPORTED_DEVICE_TYPES \
            or not _device_is_ppu(self.device):
        raise RuntimeError(
            f"gelu 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={self.device!r}（非 PPU 栈）")
    if self.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"gelu thead backend 不支持 dtype={self.dtype}；"
            f"支持: {[str(d) for d in SUPPORTED_DTYPES]}"
            "（其他 dtype 由 register 层走原生回退）")
    if approximate not in _APPROXIMATE:
        raise ValueError(
            f"gelu thead backend 不支持 approximate={approximate!r}")


def gelu_ppu(self: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """PPU 原生对照实现: 委托原生 ``aten::gelu``（redispatch 绕开本算子
    A1 注册，防递归）。
    注意: approximate 为 schema keyword-only, redispatch 亦须关键字传参
    （真机实测: 位置传参在 C++ boxed 层同样报 schema 不匹配）。
    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）: 必须用**后端键
    CUDA**——CompositeExplicitAutograd 落到 ATen 组合分解, 半精度下与
    厂商 kernel 数值分化（bf16 LN 差 0.031 同源实证）; CUDA 键与直调
    逐位一致。"""
    _validate(self, approximate)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.gelu.default.redispatch(
        keyset, self, approximate=approximate)


def gelu_backward_ppu(grad: torch.Tensor, self: torch.Tensor,
                      approximate: str = "none") -> torch.Tensor:
    """PPU gelu 反向对照: 委托原生 ``aten::gelu_backward``（同口径）。"""
    _validate(self, approximate)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.gelu_backward.default.redispatch(
        keyset, grad, self, approximate=approximate)
