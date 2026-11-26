"""Additional device checks, including bounds rejection before mutation."""
import torch
from ops.scatter_ import scatter_triton


def run(profile):
    device = profile.torch_device
    count = 0
    for dtype in (torch.int8, torch.uint8, torch.int16, torch.float64):
        dst = torch.zeros(33, dtype=dtype, device=device)
        src = torch.arange(33).to(dtype).to(device)
        index = torch.arange(32, -1, -1, device=device)
        scatter_triton(dst, 0, index, src)
        torch.testing.assert_close(dst.cpu(), src.cpu().flip(0), rtol=0, atol=0)
        count += 1
    for shape in ((128, 4096), (2, 3, 5, 7), (2, 2, 3, 4, 5)):
        dst = torch.zeros(shape, device=device)
        src = torch.rand(shape, device=device)
        index = torch.arange(shape[-1] - 1, -1, -1, device=device).expand(shape)
        scatter_triton(dst, -1, index, src)
        torch.testing.assert_close(dst.cpu(), src.cpu().flip(-1), rtol=0, atol=0)
        count += 1
    for invalid in (-1, 3):
        dst = torch.ones(3, device=device)
        index = torch.tensor([invalid], device=device)
        try:
            scatter_triton(dst, 0, index, 8.)
        except RuntimeError as exc:
            assert "out of bounds" in str(exc)
        else:
            raise AssertionError("invalid index did not raise")
        torch.testing.assert_close(dst.cpu(), torch.ones(3), rtol=0, atol=0)
        count += 1
    # Duplicate indices: test allowed winner membership, not a specific race.
    dst = torch.full((3,), -1., device=device)
    src = torch.arange(17, device=device).float()
    index = torch.zeros(17, device=device, dtype=torch.long)
    scatter_triton(dst, 0, index, src)
    assert dst[0].item() in range(17)
    assert torch.equal(dst[1:].cpu(), torch.full((2,), -1.))
    count += 1
    # Integer scalar ABI must preserve values beyond fp32's exact range.
    dst = torch.zeros(3, device=device, dtype=torch.int64)
    scatter_triton(dst, 0, torch.tensor([1], device=device), 2**50 + 3)
    assert dst[1].item() == 2**50 + 3
    return {"ok": True, "cases": count + 1}
