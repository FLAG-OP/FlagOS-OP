"""CPU-only dispatch-key selection checks for the Cambricon matmul port."""
from types import SimpleNamespace

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
