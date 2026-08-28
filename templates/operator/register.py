# 注册模板: 三条路线三选一（删除不用的）。占位名 my_op，全局替换后使用。
#
# 入口函数名必须是 register / vllm_fl_register（known-issues #8）。


# ══════════ 路线 A1: torch 算子替换（aten 宿主）══════════
def register_a1(dispatch_key: str):
    import torch
    from kernel import my_op_triton

    def aten_impl(self, *, approximate="none"):   # <TODO: 按 aten 签名>
        return my_op_triton(self, self)           # <TODO: 输入映射>

    lib = torch.library.Library("aten", "IMPL")   # 对象必须保持引用
    lib.impl("gelu", aten_impl, dispatch_key)     # <TODO: 目标 aten 算子名>
    return lib


# ══════════ 路线 A2: FlagOS 融合算子（dispatch 插件）══════════
def register(registry) -> None:                  # ← 入口名固定（#8）
    from vllm_fl.dispatch.types import (OpImpl, BackendImplKind,
                                        BackendPriority)
    from vllm_fl.dispatch.backends.base import Backend
    from kernel import my_op_triton
    from reference import my_op_reference

    class MyOpBackend(Backend):
        @property
        def name(self):
            return "myvendor"                     # <TODO: vendor 名>

        def is_available(self):
            return True

        def my_op(self, obj, x, gate):
            return my_op_triton(x, gate)

    class MyOpReference(Backend):
        @property
        def name(self):
            return "reference"

        def is_available(self):
            return True

        def my_op(self, obj, x, gate):
            return my_op_reference(x, gate)

    v, r = MyOpBackend(), MyOpReference()
    registry.register_many([
        OpImpl(op_name="my_op", impl_id="vendor.myvendor",   # <TODO: op 名>
               kind=BackendImplKind.VENDOR, fn=v.my_op,
               vendor="myvendor", priority=BackendPriority.VENDOR),
        OpImpl(op_name="my_op", impl_id="reference.torch",
               kind=BackendImplKind.REFERENCE, fn=r.my_op,
               vendor=None, priority=BackendPriority.REFERENCE),
    ])


vllm_fl_register = register


# ══════════ 路线 B: 厂商算子注册（同 A2 的 vendor 形态，见
# examples/b-fullstack/fullstack_plugin.py 完整范本）══════════
