import torch

from ops.scatter_ import register_a1
from ops.scatter_.test.cases import cases


def run(profile):
    inputs = list(cases(profile.torch_device))
    expected = [d.clone().scatter_(dim, idx, src) for _, d, dim, idx, src in inputs]
    counter = {"n": 0}
    lib = register_a1("CUDA", counter=counter, platform=profile.name)
    try:
        with torch.no_grad():
            for (name, dst, dim, index, src), ref in zip(inputs, expected):
                version = dst._version
                result = dst.scatter_(dim, index, src)
                assert result is dst
                assert dst._version == version + 1, name
                torch.testing.assert_close(dst, ref, rtol=0, atol=0, msg=name)
        assert counter["n"] == len(inputs)
        return {"ok": True, "hits": counter["n"]}
    finally:
        lib._destroy()
