# vendor 路线注册模板
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def _bind(fn, is_avail):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_avail
    return wrapper


def register_builtins(registry) -> None:
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
    from .audit_vendor import AuditVendorBackend

    backend = AuditVendorBackend()
    is_avail = backend.is_available
    registry.register_many([
        OpImpl(op_name="silu_and_mul", impl_id="vendor.audit",
               kind=BackendImplKind.VENDOR, fn=_bind(backend.silu_and_mul, is_avail),
               vendor="audit", priority=BackendPriority.VENDOR),
    ])
    logger.info("Route-B plugin: registered vendor.audit (silu_and_mul)")


register = register_builtins
vllm_fl_register = register_builtins
