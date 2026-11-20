from __future__ import annotations

import importlib

import pytest

# wt-2026-11-20-fix
# wt <liyanze1560495@hotmail.com>


class _FakeLibrary:
    registrations: list[tuple[str, str]] = []

    def __init__(self, namespace: str, kind: str):
        assert (namespace, kind) == ("aten", "IMPL")

    def impl(self, op: str, fn, dispatch_key: str) -> None:
        assert callable(fn)
        self.registrations.append((op, dispatch_key))


def _module(monkeypatch):
    import ops.sigmoid.register as register

    register = importlib.reload(register)
    _FakeLibrary.registrations.clear()
    monkeypatch.setattr(register, "_load_impl", lambda name: lambda value: value)
    monkeypatch.setattr(register.torch.library, "Library", _FakeLibrary)
    return register


def test_sigmoid_registers_canonical_and_compatibility_keys(monkeypatch):
    register = _module(monkeypatch)

    libs = register.register_a1(platform="p800-kunlunxin-gems5")

    assert len(libs) == 2
    assert register._REGISTERED_KEYS == ("AutogradCUDA", "CUDA")
    assert register._ACTIVE_IMPL == "native"
    assert _FakeLibrary.registrations == [
        ("sigmoid", "AutogradCUDA"),
        ("sigmoid", "CUDA"),
    ]


def test_sigmoid_cuda_alias_resolves_to_same_pair(monkeypatch):
    register = _module(monkeypatch)

    first = register.register_a1("CUDA", impl="triton")
    second = register.register_a1("AutogradCUDA", impl="triton")

    assert first == second
    assert register._ACTIVE_IMPL == "triton"


def test_sigmoid_rejects_unsupported_dispatch_key(monkeypatch):
    register = _module(monkeypatch)

    with pytest.raises(RuntimeError, match="AutogradCUDA"):
        register.register_a1("PrivateUse1")
