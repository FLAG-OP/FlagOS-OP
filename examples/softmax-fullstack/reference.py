# softmax 语义参考——判卷标准（fp32 计算后 cast 回输入 dtype）。
import torch
import torch.nn.functional as F


def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """最后一维 softmax。"""
    return F.softmax(x.float(), dim=-1).to(x.dtype)
