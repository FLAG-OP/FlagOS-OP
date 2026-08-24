# b-fullstack 的 dispatch 插件: 把 JIT 编译的 C++ kernel 注册为
# vendor:my-cpp（可被 VLLM_FL_PER_OP 精确钉住）。
from __future__ import annotations

import json
import os
from pathlib import Path


_COUNT_BASE = os.environ.get("MY_CPP_COUNT_FILE", "/tmp/my_cpp_counts")


def _load_kernel():
    """JIT 编译 csrc（ninja 缓存，重复加载近零耗时）。"""
    from torch.utils.cpp_extension import load

    src = Path(__file__).resolve().parents[2] / "routes/b_vendor/csrc"
    os.makedirs("/tmp/flagos_csrc_build", exist_ok=True)
    return load(name="my_vendor_ops",
                sources=[str(src / "vendor_kernel.cpp")],
                extra_cflags=["-O3"], verbose=False,
                build_directory="/tmp/flagos_csrc_build")


def _bump(op: str) -> None:
    cf = f"{_COUNT_BASE}.{os.getpid()}"   # pid 分片，消除 TP 写竞争
    counts = {}
    try:
        counts = json.load(open(cf))
    except (OSError, ValueError):
        pass
    counts[op] = counts.get(op, 0) + 1
    json.dump(counts, open(cf, "w"))


def register_builtins(registry) -> None:
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
    from vllm_fl.dispatch.backends.base import Backend

    class MyCppVendor(Backend):
        """自研 C++ kernel 的 vendor backend。"""

        @property
        def name(self): return "my-cpp"

        @property
        def vendor(self): return "my-cpp"

        def is_available(self):
            try:
                _load_kernel()
                return True
            except Exception:
                return False

        def silu_and_mul(self, obj, x):
            _bump("silu_and_mul")
            return _load_kernel().silu_and_mul(x)

    backend = MyCppVendor()
    registry.register_many([
        OpImpl(op_name="silu_and_mul", impl_id="vendor.my-cpp",
               kind=BackendImplKind.VENDOR, fn=backend.silu_and_mul,
               vendor="my-cpp", priority=BackendPriority.VENDOR),
    ])


register = register_builtins
vllm_fl_register = register_builtins
