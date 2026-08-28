# XPU 栈兼容补丁
#
# 背景（known-issues #14）: torch_xmlir 符号改写会重新导入 triton 的 xpu
# backend，产生一份**不在 sys.modules 里**的 compiler 模块实例；该实例的
# run_cmd() 引用了未 import 的 sys，导致任何 elfconv 报错都被
# "NameError: sys" 掩盖，真实错误（如 #13 的 Unsupported 符号）不可见。
#
# 本补丁通过 backends 注册表直达第二实例的 globals，注入 sys——
# 不改变编译行为，只让失败可诊断。
from __future__ import annotations


def ensure_xpu_compiler_debuggable() -> bool:
    """给被符号改写隐藏的 xpu compiler 实例注入 sys。返回是否生效。"""
    try:
        import sys
        from triton.backends import backends
        entry = backends.get("xpu")
        if entry is None:
            return False
        g = entry.compiler.make_xpubin.__globals__
        if "sys" not in g:
            g["sys"] = sys
        return True
    except Exception:
        return False
