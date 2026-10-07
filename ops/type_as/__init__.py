"""ops.type_as — aten::type_as 算子包。

包化约定（多算子同进程必用, 见 ops/README「包化导入」与
sdpa/reports/e2e_mini_llm.md 的同名冲突实证）:
  - 包态: from ops.type_as import register_a1   ← 顶层名带包前缀
  - 脚本态: sys.path.insert(0, 算子目录) 后 from register import ...
    （单算子目录内运行不变）

wt 2026-10-07-fix 补包化入口（审计发现 13 算子缺失, #14 范式补齐）
# wt <wangt635@ustc.edu.cn>
"""
from __future__ import annotations

import importlib


def _pkg_import(name: str):
    return importlib.import_module("ops.type_as." + name)


def __getattr__(name: str):
    if name == "register_a1":
        return _pkg_import("register").register_a1
    if name == "reference":
        return _pkg_import("reference").type_as_reference
    raise AttributeError(f"ops.type_as has no attribute {name!r}")


__all__ = ["register_a1", "reference"]
