"""T-head PPU (ZW810E) softmax backend — native 委托（生产路径）。

台账结论（issue #55; probes/thead_dense_p0_probe.py 真机取证, 卡6）:

- ``aten::softmax`` / ``aten::_softmax`` 在 PPU Torch-FL 栈原生存在，
  fp16/bf16/fp32 对 CPU fp32 参考误差在容差内 → **生产路径委托原生**
  （sdpa/embedding/sdpa_math 三个 thead 交付的同构结论: 归约/GEMM/
  attention 是厂商库强项）。
- softmax 是行归约算子，自研 Triton 必须处理 known-issues #15a 尾块
  归约坑（pad + 固定 BLOCK，见 triton_level.py 备选实现——按
  examples/softmax-fullstack 已落地范式，不接生产，作对照与备选）。

适用范围与 fallback 边界（issue #55 交付要求 2, 如实记录）:
  - dtype: fp32 / fp16 / bf16 全委托; 其余 dtype 走原生语义不变。
  - dim: 任意维度全委托（原生无该限制; triton 备选仅支持最后一维）。
  - 注册层负责防递归: backend 内 redispatch 到
    CompositeExplicitAutograd（sdpa 教训——A1 注册后 backend 内再调
    F.softmax/aten::softmax 会无限递归）。
"""
from __future__ import annotations

import torch

PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # PPU torch 经 CUDA 兼容层直曝 cuda:N
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_PPU_STACK = None


def _has_ppu_stack() -> bool:
    """进程级 PPU 栈识别（与 embedding/sdpa/gelu thead backend 同款）。"""
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


def _validate(self: torch.Tensor) -> None:
    if not _has_ppu_stack():
        raise RuntimeError(
            f"softmax 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 PPU 栈"
            "（误配请检查 profile，见 ops/softmax/PLATFORM.md）")
    if self.device.type not in SUPPORTED_DEVICE_TYPES \
            or not _device_is_ppu(self.device):
        raise RuntimeError(
            f"softmax 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={self.device!r}（非 PPU 栈）")
    if self.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"softmax thead backend 不支持 dtype={self.dtype}；"
            f"支持: {[str(d) for d in SUPPORTED_DTYPES]}")


def softmax_ppu(self: torch.Tensor, dim: int = -1,
                dtype: torch.dtype | None = None) -> torch.Tensor:
    """PPU 生产实现: 委托原生 ``aten::softmax.int`` 重载（redispatch 防
    递归; 本栈重载集为 int/Dimname/int_out, **无 int_dtype**——真机实测
    2026-11-25, 首版 int_dtype 假设 AttributeError 实证）。

    dtype 参数语义映射（与 _softmax 原语的关系, 平台边界如实记录）:
      dtype=None / dtype==self.dtype → _softmax(half_to_float=False)
      dtype=fp32 且 self=fp16        → _softmax(half_to_float=True)
        （厂商 h2f 仅支持 Half——bf16/fp32 输入的 h2f 报
        "conversion is supported for Half type only", 真机实证;
        bf16→fp32 走同 dtype 计算 + .to(fp32) 后处理）
      其他显式 dtype → 同 dtype 计算 + .to(dtype) 后处理

    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）: 后端键 CUDA
    直下厂商 kernel（CompositeExplicitAutograd 落组合分解, 半精度分化;
    同键注册后重入自身——RecursionError 实证）。
    """
    _validate(self)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    if dtype is None or dtype == self.dtype:
        return torch.ops.aten._softmax.default.redispatch(
            keyset, self, dim, False)
    if dtype is torch.float32 and self.dtype is torch.float16:
        return torch.ops.aten._softmax.default.redispatch(
            keyset, self, dim, True)
    out = torch.ops.aten._softmax.default.redispatch(keyset, self, dim, False)
    return out.to(dtype)


def _softmax_ppu(self: torch.Tensor, dim: int = -1,
                 half_to_float: bool = False) -> torch.Tensor:
    """PPU 原生 _softmax 委托（h2f 仅 Half, 平台边界见 softmax/PLATFORM）。"""
    _validate(self)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    if half_to_float and self.dtype is torch.float16:
        return torch.ops.aten._softmax.default.redispatch(
            keyset, self, dim, True)
    if half_to_float:
        # 厂商 h2f 仅支持 Half（真机实证）——fp32 域计算后 cast（语义
        # 等价: softmax 在 fp32 累加再转 fp32 输出）
        out = torch.ops.aten._softmax.default.redispatch(
            keyset, self.float(), dim, False)
        return out
    return torch.ops.aten._softmax.default.redispatch(
        keyset, self, dim, False)
