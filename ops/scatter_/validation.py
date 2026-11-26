"""CPU-testable metadata checks shared by the Triton entry point."""
import torch

SUPPORTED = (torch.float16, torch.bfloat16, torch.float32, torch.float64,
             torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
             torch.bool)


def validate(self, dim, index, src):
    tensor_src = isinstance(src, torch.Tensor)
    tensors = (self, index, src) if tensor_src else (self, index)
    if any(t.layout != torch.strided for t in tensors):
        raise NotImplementedError("scatter_: only strided tensors are supported")
    if any(t.device != self.device for t in tensors):
        raise RuntimeError("scatter_: all tensors must be on the same device")
    if self.dtype not in SUPPORTED:
        raise NotImplementedError(f"scatter_: unsupported dtype {self.dtype}")
    if any(t.is_conj() or t.is_neg() for t in tensors):
        raise NotImplementedError("scatter_: lazy conjugate/negative views unsupported")
    rank = max(self.ndim, 1)
    if not -rank <= dim < rank:
        raise IndexError(f"scatter_: dimension {dim} out of range for rank {rank}")
    dim %= rank
    # Native checks source dtype even for an empty index.
    if tensor_src and src.dtype != self.dtype:
        raise RuntimeError("scatter_: self and src must have the same dtype")
    if index.numel() == 0:
        return dim
    if index.dtype != torch.int64:
        raise RuntimeError("scatter_: Expected dtype int64 for index")
    if max(index.ndim, 1) != rank:
        raise RuntimeError("scatter_: index and self must have the same rank")
    shape = self.shape or (1,)
    ishape = index.shape or (1,)
    if tensor_src and max(src.ndim, 1) != rank:
        raise RuntimeError("scatter_: index and src must have the same rank")
    for d, size in enumerate(ishape):
        if d != dim and size > shape[d]:
            raise RuntimeError("scatter_: index exceeds self outside scatter dimension")
        if tensor_src and size > (src.shape or (1,))[d]:
            raise RuntimeError("scatter_: index exceeds src shape")
    # Sufficient non-overlap proof, including transposes and strided slices.
    extent = 1
    for stride, size in sorted(zip(self.stride(), self.shape)):
        if size <= 1:
            continue
        if stride < extent:
            raise NotImplementedError("scatter_: overlapping destination unsupported")
        extent += (size - 1) * stride
    for t in tensors[1:]:
        if torch._C._is_alias_of(self, t):
            raise NotImplementedError("scatter_: aliased inputs unsupported")
    if self.requires_grad or (tensor_src and src.requires_grad):
        if torch.is_grad_enabled():
            raise NotImplementedError("scatter_triton direct entry is inference-only")
    return dim
