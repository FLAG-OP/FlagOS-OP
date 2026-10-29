# A1 多算子注入: 在单个进程（含 vLLM EngineCore 子进程）内注册 FlagOS-OP
# 的多个 aten 算子。
#
# 背景: ops/<op>/register.py 的实现用 `from kernel.torch_level import ...`
# 这一「裸 kernel 包名」导入；当多个算子同时注册时，各自的 ops/<op>/kernel/
# 同名包会互相覆盖（先被导入者胜出），导致 A 算子执行到 B 算子的 kernel。
# 本模块为每个算子隔离导入其 kernel 模块，并在该算子的 aten 包装函数调用
# 期间把 sys.modules['kernel*'] 切回它自己的包，从而多算子可同时注册、
# 互不串味（详见 tests/framework_level/test_a1_aten_ops.py）。
#
# 由 injection/sitecustomize.py 在 FLAGOS_A1_ATEN_INJECT=1 时调用。
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]

# 可在真实 vLLM 进程内安全注册的算子（实测）。
# 排除项与原因:
#   empty / detach_ / result_type —— 设计上不可 A1 拦截（见各 op 的 register.py）。
#   detach —— 可独立注册，但在真实 vLLM 建模时注册会破坏
#             torch.Tensor._make_subclass(nn.Parameter 构造)，报
#             "raw Tensor object is already associated to a python object
#             of type Tensor"。与 detach_ 类似，属于进程内不可共存的算子。
_DEFAULT_OPS = [
    "type_as", "clone", "contiguous", "copy_", "dropout",
    "empty_like", "empty_strided", "item", "_local_scalar_dense",
]


def _ops() -> list[str]:
    raw = os.environ.get("FLAGOS_A1_OPS")
    if raw is None:
        return list(_DEFAULT_OPS)
    return [x for x in raw.split(",") if x]


def _purge_kernel() -> None:
    for m in [m for m in list(sys.modules)
              if m == "kernel" or m.startswith("kernel.")]:
        del sys.modules[m]


def _load_isolated(root: Path, op: str):
    opd = root / "ops" / op
    reg = opd / "register.py"
    if not reg.is_file():
        return None
    _purge_kernel()
    sys.path.insert(0, str(opd))
    try:
        spec = importlib.util.spec_from_file_location(f"_a1reg_{op}", reg)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for sub in ("kernel", "kernel.torch_level", "kernel.triton_level"):
            try:
                importlib.import_module(sub)
            except Exception:
                pass
        saved = {n: sys.modules[n] for n in list(sys.modules)
                 if n == "kernel" or n.startswith("kernel.")}
    except Exception:
        return None
    finally:
        try:
            sys.path.remove(str(opd))
        except ValueError:
            pass
    return mod, saved


def _wrap(fn, saved):
    def wrapper(*args, **kwargs):
        old = {}
        for n in saved:
            old[n] = sys.modules.get(n)
            sys.modules[n] = saved[n]
        try:
            return fn(*args, **kwargs)
        finally:
            for n, v in old.items():
                if v is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = v

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    return wrapper


def _maybe_counter(op: str):
    """若设置了 FLAGOS_C2_COUNT_DIR，则返回 (counter, dump_fn)。

    计数文件按 pid 分片（docs/testing.md: TP worker 并发写同一文件会
    低估约 35 倍）。dump 在 atexit 与每次递增后触发，进程崩溃也能留痕。
    """
    count_dir = os.environ.get("FLAGOS_C2_COUNT_DIR")
    if not count_dir:
        return None, None
    import atexit
    import json as _json
    from datetime import datetime, timezone

    cpath = Path(count_dir) / f"{op}_hits_{os.getpid()}.json"
    Path(count_dir).mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def dump():
        _aten_label = {
            "sdpa_math": "aten::_scaled_dot_product_attention_math",
            "embedding": "aten::embedding",
        }.get(op, op)
        cpath.write_text(_json.dumps({
            "pid": os.getpid(),
            "op": op,
            "aten": _aten_label,
            "hits": counter["n"],
            "dumped_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        }))

    atexit.register(dump)
    return counter, dump


# 进程内已注册算子的 counter 存档: {op: counter_dict}。
# sitecustomize 注册后，child runner 可经 get_counter(op) 取回同一份
# counter 做主进程 probe 验证（避免二次注册产生独立 counter）。
_COUNTERS: dict[str, dict] = {}

# 持有 register_a1 返回的 lib 列表——GC 后注册失效（见 register.py 注释）。
_HELD_LIBS: list = []


def get_counter(op: str):
    """返回 register_all 存档的 counter dict，未注册则 None。"""
    return _COUNTERS.get(op)


def register_all(dispatch_key: str = "PrivateUse1",
                 root: str | None = None) -> list[str]:
    """注册本仓库的 A1 aten 算子，返回成功注册的算子名列表。"""
    base = Path(root or os.environ.get("FLAGOS_TEMPLATES_ROOT") or _DEFAULT_ROOT)
    registered: list[str] = []
    for op in _ops():
        r = _load_isolated(base, op)
        if r is None:
            continue
        mod, saved = r
        for name in list(vars(mod)):
            fn = getattr(mod, name, None)
            if callable(fn) and name.endswith("_aten"):
                setattr(mod, name, _wrap(fn, saved))
        try:
            counter, _dump = _maybe_counter(op)
            # FLAGOS_A1_IMPL 可强制指定实现（如 torch），用于 C2 调试/规避
            # triton 在特定 shape 下的首轮异常。
            _impl = os.environ.get("FLAGOS_A1_IMPL") or None
            libs = None
            if counter is not None:
                try:
                    if _impl:
                        libs = mod.register_a1(
                            dispatch_key, counter=counter, impl=_impl)
                    else:
                        libs = mod.register_a1(dispatch_key, counter=counter)
                except TypeError:
                    # 个别算子 register_a1 未升级 counter/impl 参数时退回
                    libs = mod.register_a1(dispatch_key)
                # 注册成功后才存档 counter——失败时 get_counter 返回 None，
                # 避免 harness 误报 a1_counter_retrieved=true 但 forward_hits=0。
                _COUNTERS[op] = counter
            else:
                if _impl:
                    try:
                        libs = mod.register_a1(dispatch_key, impl=_impl)
                    except TypeError:
                        libs = mod.register_a1(dispatch_key)
                else:
                    libs = mod.register_a1(dispatch_key)
            if libs is not None:
                _HELD_LIBS.append(libs)
            registered.append(op)
        except Exception:
            pass
    return registered
