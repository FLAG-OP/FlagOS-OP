"""昆仑芯 P800 的 pointwise/codegen 配置。"""
from __future__ import annotations

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._kunlunxin.utils.codegen_config_utils import (
    CodeGenConfig,
)
from flag_gems.runtime.backend._kunlunxin.utils.pointwise_dynamic import (
    pointwise_dynamic,
)

# wt-2026-11-09-sigmoid-p800
# wt <liyanze1560495@hotmail.com>

PLATFORM = "p800-kunlunxin"

_CONFIG = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")], config=_CONFIG)
@triton.jit
def _sigmoid_pointwise(x):
    x_fp32 = x.to(tl.float32)
    return 1.0 / (1.0 + tl.exp(-x_fp32))


def sigmoid_kunlunxin(x):
    with torch_device_fn.device(x.device):
        return _sigmoid_pointwise(x)
