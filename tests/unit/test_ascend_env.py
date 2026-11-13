"""Unit tests for common/ascend_env.py (TASK_QUEUE opt-in argument parsing)."""
import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "common" / "ascend_env.py"
_spec = importlib.util.spec_from_file_location("ascend_env", _P)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
opt_in = _mod.task_queue_opt_in


def test_device_space_form():
    assert opt_in(["--device", "ascend910"], {})


def test_device_equals_form():
    assert opt_in(["--device=ascend910"], {})


def test_env_device():
    assert opt_in([], {"FLAGOS_DEVICE": "ascend910"})


def test_non_ascend():
    assert not opt_in(["--device", "p800-kunlunxin"], {})


def test_explicit_off():
    assert not opt_in(["--device=ascend910"],
                      {"FLAGOS_ASCEND_TASK_QUEUE": "1"})


def test_explicit_on():
    assert opt_in([], {"FLAGOS_ASCEND_TASK_QUEUE": "0"})
