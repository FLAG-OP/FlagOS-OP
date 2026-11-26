"""A1 backend-only registration for aten::scatter_ (Ascend910, torch_npu).

Preserve PyTorch autograd/in-place dispatch: register at the torch_npu
AutogradPrivateUse1 key (+ compat PrivateUse1) instead of a plain device key.
"""
import torch

_KEY_PAIRS = {"AutogradPrivateUse1": "PrivateUse1",
              "PrivateUse1": "AutogradPrivateUse1"}


def _a1_keys(dispatch_key):
    keys = [dispatch_key]
    if dispatch_key in _KEY_PAIRS:
        keys.append(_KEY_PAIRS[dispatch_key])
    return keys


def register_a1(dispatch_key="AutogradPrivateUse1", counter=None,
                impl="triton", platform=None):
    if impl != "triton":
        raise ValueError("scatter_: only impl='triton' is available")
    if platform not in (None, "ascend", "ascend910"):
        raise ValueError(f"scatter_: unsupported platform {platform!r}")
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("scatter_: Ascend NPU (torch_npu) required")

    def apply(self, dim, index, src):
        from .kernel.triton_level import scatter_triton
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return scatter_triton(self, dim, index, src)

    def value(self, dim, index, value):
        return apply(self, dim, index, value)

    lib = torch.library.Library("aten", "IMPL")
    for k in _a1_keys(dispatch_key):
        lib.impl("scatter_.src", apply, k)
        lib.impl("scatter_.value", value, k)
    return lib


register = register_a1
