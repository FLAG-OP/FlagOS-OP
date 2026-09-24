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
        import a1_aten_ops

        a1_aten_ops.register_all(
            os.environ.get("A1_DISPATCH_KEY", "PrivateUse1"),
            root=_root or None,
        )
    except Exception:
        pass
