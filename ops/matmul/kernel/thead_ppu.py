"""T-head PPU (ZW810E) matmul backend — native 委托（复用路线）。

台账结论（issue #55; probes/thead_dense_p0_probe.py 真机取证, 卡6）:

- ``aten::matmul`` 原生可用: 2D fp16 与 3D bf16 广播均通过
  （fp16 abs err 3.9e-3, bf16 err 0.0——abs 口径, GEMM 判卷用 5e-2 rel）。
- GEMM 是厂商库强项; PPU Triton `tl.dot` 精度降级（sdpa §3c: 64³
  rel≈7e-4）→ **不自研 Triton GEMM, 生产委托原生**（issue #55: 任务
  重点是确认 PPU 后端是否可复用）。
- 与在途 PR 的关系: PR #35（P800）/ PR #53（MLU590）不属于本平台交付;
  本目录在 main 上落地公共 ops/matmul 骨架, 平台范围见 PLATFORM.md /
  MERGE.md（issue #55 明确排除项条款）。
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


def _validate(self: torch.Tensor, other: torch.Tensor) -> None:
    if not _has_ppu_stack():
        raise RuntimeError(
            f"matmul 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 PPU 栈"
            "（见 ops/matmul/PLATFORM.md）")
    for name, t in [("self", self), ("other", other)]:
        if t.device.type not in SUPPORTED_DEVICE_TYPES \
                or not _device_is_ppu(t.device):
            raise RuntimeError(
                f"matmul 是 PLATFORM={PLATFORM!r} 绑定实现，"
                f"{name} 在 {t.device!r}（非 PPU 栈）")
        if t.dtype not in SUPPORTED_DTYPES:
            raise RuntimeError(
                f"matmul thead backend 不支持 {name}.dtype={t.dtype}"
                "（其他 dtype 由 register 层走原生回退）")


def matmul_ppu(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """PPU 生产实现: 委托原生 ``aten::matmul``（redispatch 防 A1 递归）。

    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）: 后端键 CUDA
    直下厂商 kernel——CompositeExplicitAutograd 落到组合分解, 累加序
    不同致半精度数值分化（bf16 LN 差 0.031 同源实证）。
    """
    _validate(self, other)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.matmul.default.redispatch(keyset, self, other)
