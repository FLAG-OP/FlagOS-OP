from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT)


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        ((2, 3, 4), (4,)),
        ((4,), (2, 4, 5)),
        ((3, 4), (2, 4, 5)),
        ((4,), (4,)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 1, 3, 4), (2, 4, 5)),
    ],
)
def test_matmul_backward_matches_native_for_reviewed_shapes(shape_a, shape_b):
    from ops.matmul.register import _matmul_backward

    torch.manual_seed(2026)
    a = torch.randn(shape_a, requires_grad=True)
    b = torch.randn(shape_b, requires_grad=True)
    output = torch.matmul(a, b)
    grad = torch.randn_like(output)
    expected = torch.autograd.grad(output, [a, b], grad)
    actual = _matmul_backward(grad, a, b.detach(), True, True)

    assert actual[0].shape == expected[0].shape
    assert actual[1].shape == expected[1].shape
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    "input_shape,weight_shape,has_bias",
    [
        ((2, 4), (4,), False),
        ((2, 5), (4, 5), True),
        ((2, 3, 5), (4, 5), True),
    ],
)
def test_linear_backward_matches_native_for_reviewed_shapes(
        input_shape, weight_shape, has_bias):
    from ops.linear.register import _linear_backward

    torch.manual_seed(2026)
    input_tensor = torch.randn(input_shape, requires_grad=True)
    weight = torch.randn(weight_shape, requires_grad=True)
    bias = torch.randn(weight_shape[0], requires_grad=True) if has_bias else None
    output = torch.nn.functional.linear(input_tensor, weight, bias)
    grad = torch.randn_like(output)
    grad_inputs = [input_tensor, weight] + ([bias] if has_bias else [])
    expected = torch.autograd.grad(output, grad_inputs, grad)
    actual = _linear_backward(
        grad, input_tensor, weight, (True, True, has_bias), has_bias
    )

    for actual_grad, expected_grad in zip(actual, expected):
        assert actual_grad is not None
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=1e-5, atol=2e-6
        )


@pytest.mark.parametrize("module_name", ["ops.layer_norm.register",
                                         "ops.native_layer_norm.register"])
def test_layer_norm_backward_keeps_frozen_weight(module_name):
    import importlib
    from unittest.mock import patch

    module = importlib.import_module(module_name)
    function = module._NativeLayerNormFn \
        if "native_layer_norm" in module_name else module._LayerNormFn
    input_tensor = torch.randn(2, 6)
    mean = torch.zeros(2, 1)
    rstd = torch.ones(2, 1)
    weight = torch.tensor([1.5, 2.0, 0.5, -1.0, 3.0, 0.25])
    bias = torch.zeros(6)
    captured = {}

    def fake_backward(*args):
        captured["weight"] = args[5]
        return torch.ones_like(input_tensor), None, None

    backend = "ops.layer_norm.kernel.thead_ppu.layer_norm_backward_ppu" \
        if "native" not in module_name else \
        "ops.native_layer_norm.kernel.thead_ppu.native_layer_norm_backward_ppu"

    class Context:
        needs_input_grad = (True, False, False, False, False, False)
        saved_tensors = (input_tensor, mean, rstd, weight, bias)
        normalized_shape = [6]

    with patch(backend, fake_backward):
        function.backward(Context(), torch.ones_like(input_tensor))

    assert captured["weight"] is weight
