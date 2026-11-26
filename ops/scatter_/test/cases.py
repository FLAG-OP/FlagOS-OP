"""Seeded exact-copy cases; unique indices avoid unspecified race winners."""
import torch


def cases(device="cpu"):
    gen = torch.Generator().manual_seed(20261125)
    for dtype in (torch.float16, torch.bfloat16, torch.float32,
                  torch.int32, torch.int64, torch.bool):
        for shape, dim in [((1031,), 0), ((7, 19), -1), ((5, 7, 13), 1)]:
            dim %= len(shape)
            dst = torch.randint(0, 9, shape, generator=gen).to(dtype).to(device)
            src = torch.randint(0, 9, shape, generator=gen).to(dtype).to(device)
            view = [1] * len(shape)
            view[dim] = shape[dim]
            index = torch.arange(shape[dim] - 1, -1, -1).reshape(view).expand(shape)
            index = index.to(device)
            yield f"{dtype}-{shape}-src", dst.clone(), dim, index, src
            yield f"{dtype}-{shape}-value", dst, dim - len(shape), index, 1
    dst = torch.full((19, 7), -1.0, device=device).t()
    src = torch.arange(19 * 9, device=device).float().reshape(19, 9).t()
    index = torch.arange(13, device=device).expand(7, 13)
    yield "transposed-partial-expanded-index", dst, 1, index, src
    dst = torch.full((7, 38), -3.0, device=device)[:, 1::2]
    index = torch.arange(19, device=device).repeat(7, 1)[:, ::2]
    src = torch.arange(140, device=device).float().reshape(7, 20)[:, ::2]
    yield "strided-storage-offset", dst, 1, index, src
    yield "scalar", torch.tensor(-1., device=device), -1, torch.tensor(0, device=device), 2.
    yield "empty", torch.ones(3, device=device), 0, torch.empty(0, dtype=torch.long, device=device), 2.
    yield "duplicate-value", torch.zeros(3, device=device), 0, torch.zeros(11, dtype=torch.long, device=device), 4.
