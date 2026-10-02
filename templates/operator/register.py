# 注册模板: 三条路线三选一（删除不用的）。占位名 my_op，全局替换后使用。
#
# 入口函数名必须是 register / vllm_fl_register（known-issues #8）。
#
# wt 2026-10-02-fix 模板升级（对齐 ops/ 实际交付范式, 见 sdpa/embedding）:
#   1. register_a1 增加 impl 与 platform 参数——平台分发（多硬件）与
#      智能路由（截流, 见 sdpa kernel/auto_dispatch.py）是一等公民
#   2. torch_npu 栈的 A1 拦截点是 AutogradPrivateUse1（PrivateUse1 永不
#      命中——sdpa 开发报告 §3.1 的 dispatch 证据链）, 按 dispatch_key
#      分发 backend
#   3. 多算子同进程: 模块内 kernel 导入走 _load_kernel_mod_mod 锚定
#      （包态优先 + 文件 fallback, 防顶层名 kernel/register 遮蔽）
# # wt <wangt635@ustc.edu.cn>


# ── 平台 backend 选择（第二平台出现时启用 kernel/backends/ 结构）──
_BACKENDS = {
    # "ascend910": _load_kernel_mod_mod("backends/ascend910") 占位
}


def _backend_for(dispatch_key: str, platform=None):
    if platform and platform in _BACKENDS:
        return _BACKENDS[platform]
    by_key = {
        "CUDA": "cuda", "AutogradCUDA": "cuda",
        "AutogradPrivateUse1": "ascend910",   # torch_npu（含 PrivateUse1）
        "PrivateUse1": "ascend910",
    }
    if dispatch_key not in by_key:
        raise RuntimeError(
            f"未知 dispatch_key={dispatch_key!r}; 已支持: {list(set(by_key.values()))}"
            "（或显式传 platform）")
    return by_key[dispatch_key]


# ══════════ 路线 A1: torch 算子替换（aten 宿主）══════════
def register_a1(dispatch_key: str = "AutogradPrivateUse1",
                counter: dict | None = None,
                impl: str = "triton",
                platform: str | None = None):
    """接管 aten::<目标算子>。

    dispatch_key: CUDA 系后端用 "AutogradCUDA"; torch_npu 栈必须
        "AutogradPrivateUse1"（见模块注释 2）
    counter: 可选 dict，调用计数（op 层拦截验证）
    impl: "triton"=纯自研（默认，验证口径） | "auto"=智能路由
        （大 S 截流原生, 须提供 kernel/auto_dispatch.py——范式见 sdpa）
    platform: 显式平台名（多平台时优先于 dispatch_key 推断）
    """
    import torch

    backend = _backend_for(dispatch_key, platform)
    _has_npu = hasattr(torch, "npu")  # torch_npu 扩展（静态检查不识别）
    if dispatch_key in ("PrivateUse1", "AutogradPrivateUse1") \
            and not (_has_npu and torch.npu.is_available()):
        raise RuntimeError(
            f"register_a1 platform={backend!r} 需要可用 NPU 设备")

    def aten_impl(query, *args, **kwargs):     # <TODO: 按 aten 签名>
        from kernel.triton_level import my_op_triton
        if counter is not None:
            counter["n"] = counter.get("n", 0) + 1
        return my_op_triton(query, *args, **kwargs)

    lib = torch.library.Library("aten", "IMPL")   # 对象必须保持引用
    lib.impl("<目标算子名>", aten_impl, dispatch_key)  # <TODO: aten 算子名>
    return lib


# ══════════ 路线 A2: FlagOS 融合算子（dispatch 插件）══════════
def register(registry) -> None:                  # ← 入口名固定（#8）
    from vllm_fl.dispatch.types import (OpImpl, BackendImplKind,
                                        BackendPriority)
    from vllm_fl.dispatch.backends.base import Backend
    from kernel.triton_level import my_op_triton
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
