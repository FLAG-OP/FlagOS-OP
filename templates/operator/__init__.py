"""ops.<算子名> — <一句话语义> 算子包。

包化约定（多算子同进程, 见 ops/README「包化导入」与
sdpa/reports/e2e_mini_llm.md 的冲突实证）:
  - 包态: from ops.<算子名> import register_a1   ← 顶层名带包前缀
  - 脚本态: sys.path.insert(0, 算子目录) 后 from register import ...
    （单算子目录内运行不变）

wt <wangt635@ustc.edu.cn>
"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    """按包内路径加载子模块。"""
    # <算子名> 与目录名一致; sed 替换时一并更新
    return importlib.import_module("ops.<算子名>." + name)


def __getattr__(name: str):
    # 惰性导出（不 eager import torch/triton）——按需增删
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").my_op_reference
    if name == "PLATFORM":
        return _pkg_import("kernel.triton_level").PLATFORM
    raise AttributeError(f"ops.<算子名> has no attribute {name!r}")


__all__ = ["register_a1", "reference", "PLATFORM"]
