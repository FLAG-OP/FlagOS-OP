"""A1 backend-only registration; preserve PyTorch autograd/in-place dispatch."""
import torch


def register_a1(dispatch_key="CUDA", counter=None, impl="triton", platform=None):
    if impl != "triton":
        raise ValueError("scatter_: only impl='triton' is available")
    if dispatch_key != "CUDA":
        raise ValueError("scatter_: register at CUDA, not AutogradCUDA")
    if platform not in (None, "p800-kunlunxin", "p800-kunlunxin-gems5"):
        raise ValueError(f"scatter_: unsupported platform {platform!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("scatter_: CUDA-compatible device required")

    def apply(self, dim, index, src):
        from .kernel.triton_level import scatter_triton
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return scatter_triton(self, dim, index, src, _increment_version=False)

    def value(self, dim, index, value):
        return apply(self, dim, index, value)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("scatter_.src", apply, dispatch_key)
    lib.impl("scatter_.value", value, dispatch_key)
    return lib


register = register_a1
