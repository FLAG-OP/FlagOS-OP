"""ops.sdpa — scaled_dot_product_attention 算子包。

包化目标（fix/sdpa-multiop-e2e 的长期方案）:
  - 多算子同进程: `from ops.sdpa import register_a1` 顶层名带包前缀,
    register/kernel 冲突根治
  - 单算子旧用法兼容: `sys.path.insert(0, ops/sdpa 目录)` 后
    `from register import register_a1` 仍工作（目录内脚本独立运行）

wt <wangt635@ustc.edu.cn>
"""
from __future__ import annotations

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent

# 单算子旧用法兼容: 目录内脚本以 `python3 test/xx.py` 运行时,
# 其相对导入（from kernel.xx import）依赖目录在 sys.path。
# 包态导入时不去注入（避免污染调用方名字空间）。
if __name__ != "ops.sdpa" and str(_HERE) not in sys.path:
    pass  # 直接脚本运行场景由脚本自身处理


def _pkg_import(name: str):
    """按包内路径加载子模块（ops.sdpa.<name>），返回模块。"""
    import importlib
    return importlib.import_module(f"ops.sdpa.{name}")


def __getattr__(name: str):
    # 惰性导出: ops.sdpa.register_a1 / sdpa_triton / PLATFORM ...
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "sdpa_triton":
        return _pkg_import("kernel.triton_level").sdpa_triton
    if name == "sdpa_auto":
        return _pkg_import("kernel.auto_dispatch").sdpa_auto
    if name == "install_patch":
        return _pkg_import("kernel.auto_dispatch").install_patch
    if name == "PLATFORM":
        return _pkg_import("kernel.triton_level").PLATFORM
    if name == "reference":
        return _pkg_import("reference").sdpa_reference
    raise AttributeError(f"ops.sdpa has no attribute {name!r}")


__all__ = ["register_a1", "sdpa_triton", "sdpa_auto", "install_patch",
           "PLATFORM", "reference"]
