"""T-head PPU (ZW810E) linear backend — native 委托（复用路线）。

台账结论（issue #55; probes/thead_dense_p0_probe.py 真机取证, 卡6）:

- ``aten::linear`` 在 PPU Torch-FL 栈原生可用（2D+bias fp16 实测通过;
  3D batched no-bias bf16 见 probe linear 项）。
- linear 的本质是 GEMM（+ bias 融合）。PPU Triton 的 `tl.dot` 精度
  实测降级到 fp16 量级（64³ rel≈7e-4, ops/sdpa/PLATFORM.md §3c），
  对 GEMM 语义（判卷线 5e-2 rel 之下、生产消费 427 模型）不可接受
  → **不自研 Triton GEMM，生产路径委托原生**（与 sdpa/embedding/
  sdpa_math 三算子的 thead 复用结论同构: GEMM/attention 是厂商库
  强项，issue #55 亦明确「任务重点是确认 PPU 后端是否可复用」）。
- torch_level（matmul+add 组合）作为 torch 级对照参与黄金/性能对比。
"""
from __future__ import annotations

import torch

PLATFORM = "thead-ppu"
SUPPORTED_DEVICE_TYPES = ("cuda",)  # PPU torch 经 CUDA 兼容层直曝 cuda:N
SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
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


def _validate(input: torch.Tensor, weight: torch.Tensor,
              bias: torch.Tensor | None) -> None:
    if not _has_ppu_stack():
        raise RuntimeError(
            f"linear 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 PPU 栈"
            "（误配请检查 profile，见 ops/linear/PLATFORM.md）")
    for t, name in [(input, "input"), (weight, "weight")] + (
            [(bias, "bias")] if bias is not None else []):
        if t.device.type not in SUPPORTED_DEVICE_TYPES \
                or not _device_is_ppu(t.device):
            raise RuntimeError(
                f"linear 是 PLATFORM={PLATFORM!r} 绑定实现，"
                f"{name} 在 {t.device!r}（非 PPU 栈）")
        if t.dtype not in SUPPORTED_DTYPES:
            raise RuntimeError(
                f"linear thead backend 不支持 {name}.dtype={t.dtype}；"
                f"支持: {[str(d) for d in SUPPORTED_DTYPES]}"
                "（其他 dtype 由 register 层走原生回退）")


def linear_ppu(input: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor | None = None) -> torch.Tensor:
    """PPU 生产实现: 委托原生 ``aten::linear``（redispatch 绕开本算子
    A1 注册，防递归; native 内部按输入形态自选 addmm 融合或 matmul 路径）。

    redispatch 键口径（PLATFORM.md, 真机实证 2026-11-25）: 必须用**后端键
    CUDA**——CompositeExplicitAutograd 落到 ATen 组合分解（matmul+add）,
    与厂商融合 kernel 的累加序不同, 半精度下数值分化; CUDA 键与直调
    逐位一致（native_layer_norm bf16 差 0.031 同源实证）。
    """
    _validate(input, weight, bias)
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    return torch.ops.aten.linear.default.redispatch(keyset, input, weight, bias)
