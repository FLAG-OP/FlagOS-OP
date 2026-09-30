"""Backend selector shared by embedding tests (per-profile)."""

def _load_backend(profile):
    """按 profile 选择 backend 模块（register._BACKENDS 同款表）。

    兼容两种加载: 直接跑 test/xx.py（test 目录在 path）与
    example.py 的 importlib 加载（OP_DIR 在 path）。
    """
    import importlib.util
    import sys
    from pathlib import Path
    op_dir = Path(__file__).resolve().parents[1]
    if str(op_dir) not in sys.path:
        sys.path.insert(0, str(op_dir))
    try:
        from register import _BACKENDS
    except ImportError:
        importlib.import_module("test_kernel_common")  # 重入自身保护
        raise
    key = "ascend910" if getattr(profile, "name", "").startswith("ascend") \
        else "p800-kunlunxin"
    return _BACKENDS[key]
