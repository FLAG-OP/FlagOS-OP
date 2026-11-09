"""sigmoid 的平台后端选择器。"""
from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>


@lru_cache(maxsize=None)
def get_impl(device_type: str):
    """返回当前运行时的专用实现；没有专用实现时返回 ``None``。"""
    if device_type == "cuda" and importlib.util.find_spec("torch_xmlir"):
        return importlib.import_module(f"{__package__}.kunlunxin").sigmoid_kunlunxin
    return None
