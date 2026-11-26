"""In-place scatter (A1); device dependencies are loaded only on invocation."""
from .register import register_a1
from .reference import scatter_reference


def scatter_triton(*args, **kwargs):
    from .kernel.triton_level import scatter_triton as impl
    return impl(*args, **kwargs)


__all__ = ["scatter_triton", "scatter_reference", "register_a1"]
