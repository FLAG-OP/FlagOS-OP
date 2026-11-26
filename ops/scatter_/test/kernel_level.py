import torch

from ops.scatter_ import scatter_triton
from ops.scatter_.test.cases import cases


def run(profile):
    count = 0
    with torch.no_grad():
        for name, dst, dim, index, src in cases(profile.torch_device):
            expected = dst.clone().scatter_(dim, index, src)
            before = (dst.data_ptr(), dst.stride(), dst.storage_offset())
            result = scatter_triton(dst, dim, index, src)
            assert result is dst, name
            assert before == (dst.data_ptr(), dst.stride(), dst.storage_offset()), name
            torch.testing.assert_close(dst, expected, rtol=0, atol=0, msg=name)
            count += 1
    from ops.scatter_.test.edge_cases import run as run_edges
    edges = run_edges(profile)
    return {"ok": True, "cases": count, "edge_cases": edges["cases"]}
