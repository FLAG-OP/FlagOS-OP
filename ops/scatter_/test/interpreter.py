"""CPU-only Triton interpreter check; not accelerator compilation evidence.

Run: TRITON_INTERPRET=1 python -m ops.scatter_.test.interpreter
"""
import os

if os.environ.get("TRITON_INTERPRET") != "1":
    raise RuntimeError("Set TRITON_INTERPRET=1; this script is CPU-only")

import torch
import triton
from ops.scatter_.kernel.triton_level import _scatter
from ops.scatter_.test.cases import cases


class Metadata(tuple):
    # XMLIR 3.0 requires constexpr.value indexing; the CPU interpreter passes
    # constexpr arguments directly rather than wrapping them in tl.constexpr.
    @property
    def value(self):
        return self


def main():
    count = 0
    for name, dst, dim, index, src in cases():
        # bfloat16 interpreter uses NumPy, which has no native bf16 dtype.
        if dst.dtype == torch.bfloat16 or index.numel() == 0:
            continue
        expected = dst.clone().scatter_(dim, index, src)
        dim %= max(dst.ndim, 1)
        tensor_src = isinstance(src, torch.Tensor)
        shape = tuple(index.shape) or (1,)
        ss = (tuple(src.stride()) or (1,)) if tensor_src else (0,) * len(shape)
        args = (dst, index, src if tensor_src else dst, 0 if tensor_src else src,
                index.numel(), Metadata(shape), Metadata(dst.stride() or (1,)),
                Metadata(index.stride() or (1,)), Metadata(ss), dim,
                (dst.shape or (1,))[dim], tensor_src)
        full, tail = divmod(index.numel(), 256)
        if full:
            _scatter[(full,)](*args, BLOCK=256)
        if tail:
            _scatter[(tail,)](*args, BLOCK=1, START=full * 256)
        torch.testing.assert_close(dst, expected, rtol=0, atol=0, msg=name)
        count += 1
    print(f"Triton CPU interpreter: {count} PASS (not device validation)")


if __name__ == "__main__":
    main()
