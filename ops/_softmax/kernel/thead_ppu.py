"""T-head PPU (ZW810E) _softmax backend — 原语 native 委托（生产路径）。

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
            f"_softmax 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 PPU 栈"
            "（误配请检查 profile，见 ops/_softmax/PLATFORM.md）")
    if self.device.type not in SUPPORTED_DEVICE_TYPES \
            or not _device_is_ppu(self.device):
        raise RuntimeError(
            f"_softmax 是 PLATFORM={PLATFORM!r} 绑定实现，"
            f"收到 device={self.device!r}（非 PPU 栈）")
    if self.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"_softmax thead backend 不支持 dtype={self.dtype}；"
            f"支持: {[str(d) for d in SUPPORTED_DTYPES]}")


def _softmax_ppu(self: torch.Tensor, dim: int = -1,
                 half_to_float: bool = False) -> torch.Tensor:
    """PPU 生产实现: 委托原生 ``aten::_softmax``（后端键 CUDA redispatch
    ——CompositeExplicitAutograd 落组合分解半精度分化（不可用）; 同键
    （CUDA）注册后重入自身（RecursionError）——本算子注册 AutogradCUDA）。

    h2f 边界（真机实证 2026-11-25）: 厂商 h2f **仅支持 Half 输入**
    （bf16/fp32 报 "conversion is supported for Half type only"）——
    bf16/fp32 的 h2f 回退 fp32 域计算 + cast（softmax 输出有界, 与
    h2f 数学等价; 平台边界记录于 PLATFORM.md）。
    """
    _validate(self)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    if half_to_float and self.dtype is torch.float16:
        return torch.ops.aten._softmax.default.redispatch(
            keyset, self, dim, True)
    if half_to_float:
        out = torch.ops.aten._softmax.default.redispatch(
            keyset, self.float(), dim, False)
        return out
    return torch.ops.aten._softmax.default.redispatch(
        keyset, self, dim, False)
