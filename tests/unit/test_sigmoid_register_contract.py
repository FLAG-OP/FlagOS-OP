from __future__ import annotations

import importlib

import pytest

# wt-2026-11-20-fix
# wt <liyanze1560495@hotmail.com>


class _FakeLibrary:
    registrations: list[tuple[str, str]] = []
    fail_on: str | None = None

    def __init__(self, namespace: str, kind: str):
        assert (namespace, kind) == ("aten", "IMPL")

    def impl(self, op: str, fn, dispatch_key: str) -> None:
        assert callable(fn)
        if dispatch_key == self.fail_on:
            raise RuntimeError(f"fake registration failure: {dispatch_key}")
        self.registrations.append((op, dispatch_key))


def _module(monkeypatch):
    import ops.sigmoid.register as register

    register = importlib.reload(register)
    _FakeLibrary.registrations.clear()
    _FakeLibrary.fail_on = None
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


def test_independent_equal_counters_are_both_updated(monkeypatch):
    register = _module(monkeypatch)
    first = {}
    second = {}

    register.register_a1(counter=first)
    register.register_a1(counter=second)
    register._bump()

    assert first == {"n": 1}
    assert second == {"n": 1}
    assert len(register._COUNTERS) == 2
    assert register._COUNTERS[0] is first
    assert register._COUNTERS[1] is second


def test_reusing_same_counter_object_does_not_double_count(monkeypatch):
    register = _module(monkeypatch)
    counter = {}

    register.register_a1(counter=counter)
    register.register_a1("CUDA", counter=counter)
    register._bump()

    assert counter == {"n": 1}
    assert len(register._COUNTERS) == 1
    assert register._COUNTERS[0] is counter


def test_existing_registration_conflict_does_not_save_counter(monkeypatch):
    register = _module(monkeypatch)
    accepted = {}
    rejected = {}
    register.register_a1(impl="native", counter=accepted)

    with pytest.raises(RuntimeError, match="cannot switch"):
        register.register_a1(impl="triton", counter=rejected)

    assert len(register._COUNTERS) == 1
    assert register._COUNTERS[0] is accepted
    assert all(saved is not rejected for saved in register._COUNTERS)


def test_library_failure_leaves_registration_and_counter_state_clean(
        monkeypatch):
    register = _module(monkeypatch)
    counter = {}
    _FakeLibrary.fail_on = "CUDA"

    with pytest.raises(RuntimeError, match="fake registration failure"):
        register.register_a1(counter=counter)

    assert register._COUNTERS == []
    assert register._LIBS == {}
    assert register._REGISTERED_KEYS == ()
    assert register._ACTIVE_IMPL == "native"
