from __future__ import annotations

import importlib
import inspect

import pytest
import torch


ISSUE_19_OPS = (
    "_local_scalar_dense",
    "clone",
    "contiguous",
    "copy_",
    "detach",
    "detach_",
    "dropout",
    "empty",
    "empty_like",
    "empty_strided",
    "item",
    "result_type",
    "type_as",
)

REGISTERABLE = {
    "_local_scalar_dense": "lsd_aten",
    "clone": "clone_aten",
    "contiguous": "contiguous_aten",
    "copy_": "copy_aten",
    "detach": "detach_aten",
    "dropout": "dropout_aten",
    "empty_like": "empty_like_aten",
    "empty_strided": "empty_strided_aten",
    "item": "item_aten",
    "type_as": "type_as_aten",
}


@pytest.fixture()
def modules():
    return {
        name: importlib.import_module(f"ops.{name}.register")
        for name in ISSUE_19_OPS
    }


def test_issue_19_register_a1_signatures(modules):
    for name, module in modules.items():
        parameters = inspect.signature(module.register_a1).parameters
        assert list(parameters) == ["dispatch_key", "counter", "platform"], name
        assert parameters["dispatch_key"].default == "AutogradPrivateUse1", name
        assert parameters["counter"].default is None, name
        assert parameters["platform"].default is None, name


def test_issue_19_rejects_unknown_platform_before_registration(modules):
    for name, module in modules.items():
        with pytest.raises(RuntimeError, match="未知 platform"):
            module.register_a1(
                "PrivateUse1", counter={"n": 0}, platform="unsupported"
            )


def test_issue_19_counter_is_captured_by_registered_function(modules, monkeypatch):
    class FakeLibrary:
        def __init__(self, namespace, kind):
            assert namespace == "aten"
            assert kind == "IMPL"
            self.impl_fn = None
            self.dispatch_key = None

        def impl(self, op_name, function, dispatch_key):
            self.op_name = op_name
            self.impl_fn = function
            self.dispatch_key = dispatch_key

    fake_library = FakeLibrary
    monkeypatch.setattr(torch.library, "Library", fake_library)

    for name, aten_name in REGISTERABLE.items():
        module = modules[name]
        sentinel = object()
        monkeypatch.setattr(
            module, aten_name, lambda *args, **kwargs: sentinel
        )
        counter = {"n": 0}
        library = module.register_a1(
            "PrivateUse1", counter=counter, platform="cambricon"
        )
        assert isinstance(library, FakeLibrary), name
        assert library.dispatch_key == "PrivateUse1", name
        assert counter == {"n": 0}, name
        assert library.impl_fn(*()) is sentinel, name
        assert counter == {"n": 1}, name


def test_issue_19_registerable_ops_use_package_relative_kernel_imports():
    import ast
    from pathlib import Path

    for name in REGISTERABLE:
        source = (Path("ops") / name / "register.py").read_text()
        assert "from .kernel." in source, name
