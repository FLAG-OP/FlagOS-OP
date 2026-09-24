# torch 级实现: redispatch 到 CompositeExplicitAutograd（避免注册后自递归）。
from __future__ import annotations

import torch

_CE = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd)


def detach_torch(self):
    return torch.ops.aten.detach.default.redispatch(_CE, self)
