# 厂商硬件语言 kernel 直测接口（泛化层）。
#
# 每芯片在设备 profile 的 vendor_kernels 段声明可直测的 kernel 清单，
# 本模块提供统一的 KernelSpec 加载 / 调用 / 哨兵健全性检查。
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


# ============ PyTorch 语义参考库（与 routes/ 的参考实现保持一致） ============

def ref_gelu_and_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """GELU(tanh 近似) * gate，与 A2 路线 gelu_and_mul 同语义。"""
    xf = x.float()
    inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + torch.tanh(inner)) * gate.float()).to(x.dtype)


def ref_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


SEMANTIC_REFS: dict[str, Callable] = {
    "gelu_and_mul": ref_gelu_and_mul,
    "silu": ref_silu,
    "relu": torch.relu,
}


# ============ 常见厂商 kernel 签名适配器 ============

def call_gelu_tanh_and_mul(fn, *parts, out=None):
    """xtorch_ops.gelu_tanh_and_mul(input: [..., 2d]) -> Tensor。

    parts = (x, gate) 两个张量 → 在最后一维拼回 [..., 2d]。
    """
    x, gate = parts
    full = torch.cat([x, gate], dim=-1)
    return fn(full)


def call_silu(fn, *parts, out=None):
    return fn(parts[0])


def call_two_tensor(fn, *parts, out=None):
    """通用两张量输入: fn(x, gate) -> Tensor（自研 C++ 模板等）。"""
    return fn(*parts)


def call_out_param(fn, *parts, out=None):
    """通用 out 参数模式: fn(x, out)。"""
    if out is None:
        raise ValueError("out_param 模式必须提供 out")
    fn(parts[0], out)
    return out


SIGNATURE_ADAPTERS: dict[str, Callable] = {
    "gelu_tanh_and_mul": call_gelu_tanh_and_mul,
    "silu": call_silu,
    "two_tensor": call_two_tensor,
    "out_param": call_out_param,
}


@dataclass
class KernelSpec:
    """一个可直测的硬件语言 kernel 的完整描述。"""

    name: str                      # 如 "xtorch_ops.gelu_tanh_and_mul"
    op: str                        # 语义锚点（SEMANTIC_REFS 键，如 gelu_and_mul）
    out_mode: str = "return"       # return | out_param
    adapter: str = "gelu_tanh_and_mul"  # SIGNATURE_ADAPTERS 键
    n_inputs: int = 2              # 语义参考的输入张量数（gelu_and_mul=2, silu=1）
    expected_ok: bool = True       # False = 已知坏例（哨兵检查应能抓出）
    build: Optional[str] = None    # "csrc" = 需要 JIT 编译
    known_issue: str = ""          # 坏例的问题说明
    raw: dict = field(default_factory=dict)

    # ---------- 派生 ----------
    @property
    def module_name(self) -> str:
        return self.name.rsplit(".", 1)[0]

    @property
    def func_name(self) -> str:
        return self.name.rsplit(".", 1)[1]

    def load_callable(self) -> Callable:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.func_name)

    def call(self, fn, *parts, out=None):
        return SIGNATURE_ADAPTERS[self.adapter](fn, *parts, out=out)

    def make_inputs(self, shape, dtype, device) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(hash(self.name) & 0xFFFF)
        if self.n_inputs == 2:
            return (torch.randn(*shape, dtype=dtype, device=device) * 2,
                    torch.randn(*shape, dtype=dtype, device=device))
        return (torch.randn(*shape, dtype=dtype, device=device) * 2,)


def load_kernel_specs(profile) -> list[KernelSpec]:
    """从设备 profile 的 vendor_kernels 段加载 kernel 清单。"""
    specs = []
    for entry in (profile.raw.get("vendor_kernels") or []):
        if not entry.get("enabled", True):
            continue
        specs.append(KernelSpec(
            name=entry["name"],
            op=entry.get("op", ""),
            out_mode=entry.get("out_mode", "return"),
            adapter=entry.get("adapter", "out_param"),
            n_inputs=entry.get("n_inputs", 2),
            expected_ok=entry.get("expected_ok", True),
            build=entry.get("build"),
            known_issue=entry.get("known_issue", ""),
            raw=entry,
        ))
    return specs


# ============ 哨兵健全性检查（通用化的 swiglu 型 bug 检测） ============

def sentinel_check(spec: KernelSpec, fn, device: str,
                   shape=(64, 1024), dtype=torch.bfloat16) -> dict:
    """检测 kernel 是否真实产出（针对"不写输出"类 bug）。

    return 模式: 同输入调用两次，结果应一致且随输入变化。
    out_param 模式: 哨兵预填 out → 调用 → 检查哨兵被覆写。
    """
    parts = spec.make_inputs(shape, dtype, device)

    if spec.out_mode == "out_param":
        x = parts[0]
        d = x.shape[-1] // 2 if spec.n_inputs == 2 else x.shape[-1]
        out = torch.full((*x.shape[:-1], d), 12345.0,
                         dtype=x.dtype, device=x.device)
        try:
            spec.call(fn, *parts, out=out)
            torch.cuda.synchronize()
        except Exception as e:
            return {"ok": False, "mode": "out_param",
                    "detail": f"调用异常: {e}"}
        untouched = int((out == 12345.0).sum().item())
        return {"ok": untouched < out.numel(), "mode": "out_param",
                "detail": (f"哨兵 {untouched}/{out.numel()} 未覆写"
                           if untouched else "哨兵全部覆写 ✓")}

    # return 模式: 确定性 + 输入敏感性
    try:
        o1 = spec.call(fn, *parts)
        o2 = spec.call(fn, *parts)
        torch.cuda.synchronize()
    except Exception as e:
        return {"ok": False, "mode": "return", "detail": f"调用异常: {e}"}
    deterministic = torch.equal(o1, o2)
    parts2 = tuple(p + 1.0 for p in parts)
    o3 = spec.call(fn, *parts2)
    sensitive = not torch.equal(o1, o3)
    ok = deterministic and sensitive
    return {"ok": ok, "mode": "return",
            "detail": f"确定性={'✓' if deterministic else '✗'} "
                      f"输入敏感={'✓' if sensitive else '✗'}"}
