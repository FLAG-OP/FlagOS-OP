"""T-head PPU (ZW810E) spmm backend — index_add 组合（自建, 原生已证损坏）。

台账结论（issue #57; probes/thead_dense_p0_probe.py 真机取证, 卡6,
2026-11-25）:

- **原生 CSR SpMM 在 PPU 栈不可用作生产路径**（三连取证）:
  1. ``torch.sparse.mm`` fp32 暖进程可用（err 8.6e-6）但冷进程首发报
     ``cusparseSpMM_bufferSize: function is not implemented``;
  2. fp16 路径稳定报同一 cusparse 缺口;
  3. 组合探针进程级 **C++ abort**（returncode -6, terminate called）——
     不可捕获, 独立子进程隔离取证（probe --isolated）。
- 结论按 issue #57「存在可复现的 PPU 缺口」分支处理: 自建实现。
  生产路径 = index_add 组合（gather + scaled accum, 全部落在厂商强项
  的原生逐点/归约算子上, 与 Torch-FL 训练链路的 index_add 回退同构,
  issue #57 背景段即记载该回退为基线）。
- Triton CSR row-kernel 备选: 需 #15a 尾块归约防护且 PPU tl.dot 精度
  限制不适用（SpMM 无 dot）——保留为后续探针, 不在本轮范围。
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


def _validate(csr: torch.Tensor, dense: torch.Tensor) -> None:
    if not _has_ppu_stack():
        raise RuntimeError(
            f"spmm 是 PLATFORM={PLATFORM!r} 绑定实现，当前环境无 PPU 栈"
            "（见 ops/spmm/PLATFORM.md）")
    if not csr.is_sparse_csr:
        raise TypeError("spmm expects a CSR sparse tensor as self")
    for name, t in [("csr", csr), ("dense", dense)]:
        if t.device.type not in SUPPORTED_DEVICE_TYPES \
                or not _device_is_ppu(t.device):
            raise RuntimeError(
                f"spmm 是 PLATFORM={PLATFORM!r} 绑定实现，"
                f"{name} 在 {t.device!r}（非 PPU 栈）")
    if dense.dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"spmm thead backend 不支持 dense.dtype={dense.dtype}；"
            f"支持: {[str(d) for d in SUPPORTED_DTYPES]}")
    if csr.shape[1] != dense.shape[0]:
        raise ValueError(
            f"spmm shape mismatch: csr ({csr.shape[0]},{csr.shape[1]}) @ "
            f"dense ({dense.shape[0]},{dense.shape[1]})")


def _csr_rows(csr: torch.Tensor):
    """CSR → (rows, cols, vals) COO 形式（rows 按 nnz 展开）。"""
    crow = csr.crow_indices()
    col = csr.col_indices()
    val = csr.values()
    m = crow.shape[0] - 1
    rows = torch.repeat_interleave(
        torch.arange(m, device=val.device), crow[1:] - crow[:-1])
    return rows, col, val


def spmm_ppu(csr: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """PPU 生产实现: index_add 组合（gather + scaled accumulation）。

    out[i, j] = Σ_r∈row_i vals[r] · dense[col_r, j]——重复列索引自然
    累加（CSR 语义）, 空行零贡献（输出零行）。
    累加精度: **fp32 累加后 cast 回值 dtype**（与 CUDA 稀疏惯例一致:
    torch 原生 half SpMM 内部 fp32 累加; bf16 逐加累加的序噪声实测
    0.0625, fp32 累加消除——判卷见 test/kernel_level.py 5e-2 rel 口径）。
    """
    _validate(csr, dense)
    rows, col, val = _csr_rows(csr)
    m = csr.shape[0]
    n = dense.shape[1]
    out = torch.zeros(m, n, dtype=torch.float32, device=val.device)
    out.index_add_(0, rows,
                   val.float().unsqueeze(1) * dense.float().index_select(0, col))
    return out.to(val.dtype)


def spmm_backward_ppu(csr: torch.Tensor, dense: torch.Tensor,
                      grad_out: torch.Tensor):
    """PPU 反向: (d_vals, dB)——COO gather/index_add 组合（fp32 累加,
    口径同 spmm_ppu）。"""
    _validate(csr, dense)
    rows, col, val = _csr_rows(csr)
    d_vals = (grad_out.index_select(0, rows)
              * dense.index_select(0, col)).sum(-1)
    dB = torch.zeros(dense.shape, dtype=torch.float32, device=dense.device)
    dB.index_add_(0, col,
                  val.float().unsqueeze(1)
                  * grad_out.float().index_select(0, rows))
    return d_vals.to(val.dtype), dB.to(dense.dtype)
