"""CPU-only dispatch-key and registration checks for the Cambricon matmul port."""
from types import SimpleNamespace

import ops.matmul.register as register
from ops.matmul.register import dispatch_key_for_profile


def profile(vendor, dispatch_key):
    return SimpleNamespace(vendor=vendor, dispatch_key=dispatch_key)


def test_cambricon_private_use_is_promoted_to_autograd_key():
    assert dispatch_key_for_profile(
        profile("cambricon", "PrivateUse1")
    ) == "AutogradPrivateUse1"
    assert dispatch_key_for_profile(
        profile("cambricon", "AutogradPrivateUse1")
    ) == "AutogradPrivateUse1"


def test_unrelated_dispatch_keys_are_preserved():
    assert dispatch_key_for_profile(
        profile("cambricon", "CPU")
    ) == "CPU"
    assert dispatch_key_for_profile(
        profile("kunlunxin", "CUDA")
    ) == "AutogradCUDA"


class _FakeLibrary:
    calls = []

    def __init__(self, namespace, kind):
        self.namespace = namespace
        self.kind = kind
        self.implementations = {}

    def impl(self, operator, implementation, dispatch_key):
        assert (operator, self.namespace, self.kind) == (
            "matmul", "aten", "IMPL"
        )
        self.implementations[dispatch_key] = implementation
        self.calls.append(dispatch_key)


def _forward(left, right):
    return left, right


def _backward(*args):
    return None


def test_autograd_key_also_registers_inference_backend(monkeypatch):
    _FakeLibrary.calls = []
    backend = SimpleNamespace(
        matmul_triton=_forward,
        matmul_backward_triton=_backward,
    )
    monkeypatch.setattr(register, "import_module", lambda _: backend)
    monkeypatch.setattr(register.torch.library, "Library", _FakeLibrary)

    counter = {}
    libraries = register.register_a1("AutogradPrivateUse1", counter=counter)

    assert len(libraries) == 2
    assert _FakeLibrary.calls == ["AutogradPrivateUse1", "PrivateUse1"]
    autograd_library, backend_library = libraries
    assert set(autograd_library.implementations) == {"AutogradPrivateUse1"}
    assert set(backend_library.implementations) == {"PrivateUse1"}

    left = SimpleNamespace(shape=(2, 3))
    right = SimpleNamespace(shape=(3, 4))
    inference_impl = backend_library.implementations["PrivateUse1"]
    assert inference_impl(left, right) == (left, right)
    assert counter == {"forward": 1}


def test_non_private_autograd_key_keeps_single_registration(monkeypatch):
    _FakeLibrary.calls = []
    backend = SimpleNamespace(
        matmul_triton=_forward,
        matmul_backward_triton=_backward,
    )
    monkeypatch.setattr(register, "import_module", lambda _: backend)
    monkeypatch.setattr(register.torch.library, "Library", _FakeLibrary)

    library = register.register_a1("AutogradCUDA")

    assert isinstance(library, _FakeLibrary)
    assert _FakeLibrary.calls == ["AutogradCUDA"]
