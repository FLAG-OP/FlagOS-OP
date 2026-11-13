"""Ascend 前置环境判定（无 torch 依赖，便于单测）。

TASK_QUEUE_ENABLE=0 必须在 torch_npu 初始化前生效，故在 common/device.py
import torch 之前调用本模块。
"""
from __future__ import annotations


def _hint(argv, env) -> str:
    parts = [env.get("FLAGOS_DEVICE", ""), env.get("FLAGOS_PROFILE", "")]
    parts.extend(argv or [])
    return " ".join(parts).lower()


def ascend_selected(argv=None, env=None) -> bool:
    """argv/env 是否显式选择了 Ascend（支持 `--device ascend910`
    与 `--device=ascend910` 两种写法）。"""
    return "ascend" in _hint(argv, env or {})


def task_queue_opt_in(argv=None, env=None) -> bool:
    """是否应关闭 torch_npu task queue（TASK_QUEUE_ENABLE=0）。

    - FLAGOS_ASCEND_TASK_QUEUE=0 显式开启；
    - FLAGOS_ASCEND_TASK_QUEUE=1 显式关闭；
    - 否则：显式选择 Ascend 时开启（含空格/等号两种 --device 写法）。
    """
    env = env or {}
    opt = env.get("FLAGOS_ASCEND_TASK_QUEUE")
    if opt == "1":
        return False
    if opt == "0":
        return True
    return ascend_selected(argv, env)
