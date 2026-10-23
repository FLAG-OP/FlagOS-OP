"""CPU-only checks for the matmul package and backend dependency boundary."""
# tyf-2026-10-23-fix
# tyf <tyanfeng@mail.ustc.edu.cn>
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["package", "standalone"])
def test_matmul_registration_without_hardware_dependencies(mode):
    # A separate process isolates global ATen registration and import state.
    script = r'''
import importlib
import sys
import torch

class NoHardwareImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'triton', 'flag_gems'}:
            raise ModuleNotFoundError('blocked hardware dependency: ' + fullname)

sys.meta_path.insert(0, NoHardwareImports())
if sys.argv[1] == 'standalone':
    sys.path.insert(0, 'ops/matmul')
    module = importlib.import_module('register')
else:
    module = importlib.import_module('ops.matmul')
assert callable(module.register_a1)
assert 'triton' not in sys.modules and 'flag_gems' not in sys.modules

# The torch backend must remain usable without importing Triton. Compare both
# broadcast gradients against native before installing the registration.
torch.manual_seed(7)
a = torch.randn(2, 1, 3, 4, requires_grad=True)
b = torch.randn(1, 5, 4, 6, requires_grad=True)
expected = torch.matmul(a, b)
grad = torch.randn_like(expected)
expected_grads = torch.autograd.grad(expected, (a, b), grad)
counter = {}
lib = module.register_a1('AutogradCPU', counter=counter, backend='torch')
out = torch.matmul(a, b)
actual_grads = torch.autograd.grad(out, (a, b), grad)
torch.testing.assert_close(out, expected)
for actual, reference in zip(actual_grads, expected_grads):
    torch.testing.assert_close(actual, reference)
assert counter['forward'] == 1

# Selecting Triton must report the actual missing dependency, not "kernel".
try:
    module.register_a1('AutogradCPU', backend='triton')
except ModuleNotFoundError as exc:
    assert 'blocked hardware dependency: triton' in str(exc), str(exc)
else:
    raise AssertionError('Triton backend ignored missing dependency')
print('PASS', sys.argv[1])
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, mode], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
