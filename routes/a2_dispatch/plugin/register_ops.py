# vllm_fl dispatch 插件注册模板
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_builtins(registry) -> None:
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
    from vllm_fl.dispatch.backends.base import Backend
    from .kernels import (
        gelu_and_mul_triton, gelu_and_mul_reference, silu_and_mul_triton_counted,
    )

    class TritonCustomBackend(Backend):
        @property
        def name(self): return "flagos"

        def is_available(self):
            try:
                import triton, flag_gems  # noqa: F401
                return True
            except ImportError:
                return False

        def gelu_and_mul(self, obj, x, gate):
            return gelu_and_mul_triton(x, gate)

    class ReferenceCustomBackend(Backend):
        @property
        def name(self): return "reference"

        def is_available(self):
            return True

        def gelu_and_mul(self, obj, x, gate):
            return gelu_and_mul_reference(x, gate)

    class TritonTemplateVendor(Backend):
        """以 vendor 身份注册的 Triton silu_and_mul。

        PER_OP 支持 vendor:<name> 精确钉住，故框架级测试用它承载
        Triton 自定义实现（避免与内置 default.flagos impl_id 冲突）。
        """

        @property
        def name(self): return "triton-template"

        @property
        def vendor(self): return "triton-template"

        def is_available(self):
            try:
                import triton, flag_gems  # noqa: F401
                return True
            except ImportError:
                return False

        def silu_and_mul(self, obj, x):
            return silu_and_mul_triton_counted(x)

    fl = TritonCustomBackend()
    ref = ReferenceCustomBackend()
    tt = TritonTemplateVendor()
    registry.register_many([
        OpImpl(op_name="gelu_and_mul", impl_id="default.flagos",
               kind=BackendImplKind.DEFAULT, fn=fl.gelu_and_mul,
               vendor=None, priority=BackendPriority.DEFAULT),
        OpImpl(op_name="gelu_and_mul", impl_id="reference.torch",
               kind=BackendImplKind.REFERENCE, fn=ref.gelu_and_mul,
               vendor=None, priority=BackendPriority.REFERENCE),
        OpImpl(op_name="silu_and_mul", impl_id="vendor.triton-template",
               kind=BackendImplKind.VENDOR, fn=tt.silu_and_mul,
               vendor="triton-template", priority=BackendPriority.VENDOR),
    ])
    logger.info("A2 plugin: gelu_and_mul(flagos+reference), "
                "silu_and_mul(vendor.triton-template)")


# ⚠️ discovery.py 实际查找 `register` 或 `vllm_fl_register`
register = register_builtins
vllm_fl_register = register_builtins
