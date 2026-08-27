"""KernelSpec 签名适配器与哨兵检查单测（CPU 即可，无 GPU）。"""
import pytest

torch = pytest.importorskip("torch")

from common.kernel_spec import (KernelSpec, SIGNATURE_ADAPTERS,
                                sentinel_check)


def test_adapter_two_tensor():
    fn = lambda x, gate: x * gate
    out = SIGNATURE_ADAPTERS["two_tensor"](
        fn, torch.ones(4), torch.full((4,), 3.0))
    assert torch.equal(out, torch.full((4,), 3.0))


def test_adapter_out_param():
    box = {}

    def fn(x, out):
        box["called"] = True
        out.copy_(x * 2)

    x = torch.ones(4)
    out = SIGNATURE_ADAPTERS["out_param"](fn, x, out=torch.empty(4))
    assert box["called"] and torch.equal(out, torch.full((4,), 2.0))


def test_make_inputs_shape_dtype():
    spec = KernelSpec(name="m.f", op="silu", n_inputs=2)
    a, b = spec.make_inputs((8, 16), torch.bfloat16, "cpu")
    assert a.shape == b.shape == (8, 16)
    assert a.dtype == b.dtype == torch.bfloat16


def test_sentinel_return_mode_cpu():
    spec = KernelSpec(name="m.f", op="silu", n_inputs=1,
                      out_mode="return", adapter="silu")
    good = lambda x: torch.sigmoid(x) * x
    r = sentinel_check(spec, good, "cpu")
    assert r["ok"], r

    bad = lambda x: torch.ones_like(x)          # 输入不敏感
    r = sentinel_check(spec, bad, "cpu")
    assert not r["ok"]
