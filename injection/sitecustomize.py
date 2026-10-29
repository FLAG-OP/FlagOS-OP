# A1 框架级测试的跨进程 aten 注入桥。
#
# vLLM v1 的模型前向跑在 EngineCore/Worker 子进程中，主进程的
# torch.library 注册不会传播。把本目录加入 PYTHONPATH 后，每个
# Python 进程（含 spawn 子进程）启动时都会执行 sitecustomize，
# 从而在每个进程内完成 aten::silu 注册。
import os

if os.environ.get("FLAGOS_TEMPLATES_A1_INJECT") == "1":
    try:
        import sys

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)

        from routes.a1_aten import register_aten

        register_aten.register_silu_aten(
            os.environ.get("A1_DISPATCH_KEY", "CUDA")
        )
    except Exception:
        # 注入失败不能影响宿主进程启动（如 tokenizer 等轻量子进程）
        pass

# FlagOS-OP 自研 aten 算子的多算子注入（真实 vLLM 框架级验证用）。
# 与上面的 silu 注入相互独立，可同时开启。
if os.environ.get("FLAGOS_A1_ATEN_INJECT") == "1":
    try:
        import sys

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)
        # sitecustomize 启动时 torch_npu 尚未 import，部分算子的
        # register_a1 会探测 torch.npu.is_available()（如 sdpa_math）。
        # 此处先确保 torch_npu 已加载，再注册——spawn 子进程启动时
        # NPU 尚未被 vLLM 初始化，import torch_npu 本身是安全的。
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401
        except Exception:
            pass
        import a1_aten_ops

        a1_aten_ops.register_all(
            os.environ.get("A1_DISPATCH_KEY", "PrivateUse1"),
            root=_root or None,
        )
    except Exception:
        pass

# FlagOS C2 引擎侧 attention 后端补丁（sdpa_math × vLLM-Ascend）。
# FLAGOS_SDPA_MATH_ATTN=1 时，把 NPUPlatform.get_attn_backend_cls 补丁为
# 返回 FlagOSMathAttentionBackend——真实前向走 aten math 后端（A1 拦截目标），
# 突破 known-issues #19 的 routing boundary。
# spawn 子进程启动时 sitecustomize 先于 torch_npu 执行；此处直接尝试
# import vllm_ascend.platform（PYTHONPATH 已含源码树）并补丁。
# 若 import 失败则静默，child runner 的兜底补丁会在主进程重试。
if os.environ.get("FLAGOS_SDPA_MATH_ATTN") == "1":
    try:
        import sys as _sys

        _root = os.environ.get("FLAGOS_TEMPLATES_ROOT", "")
        if _root and _root not in _sys.path:
            _sys.path.insert(0, _root)
        from ops.sdpa_math.engine.flagos_math_attn import (
            patch_npu_platform_get_attn_backend_cls,
        )

        patch_npu_platform_get_attn_backend_cls()
    except Exception:
        pass
