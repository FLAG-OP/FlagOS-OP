# result_type 语义参考实现。
#
# 语义 (aten::result_type(...) -> ScalarType):
#   两输入在 PyTorch 类型提升规则下的公共 dtype。纯元数据，无设备码。
from __future__ import annotations

import torch


def result_type_reference(a, b):
    return torch.result_type(a, b)
