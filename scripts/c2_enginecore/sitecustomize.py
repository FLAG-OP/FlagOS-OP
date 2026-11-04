"""C2 EngineCore collection hook (Ascend A1 aten, vllm-plugin-fl stack).

Loaded automatically (as `sitecustomize`) in every Python process when this
directory is on PYTHONPATH. Only acts when FLAGOS_C2_RUN_ID is set.

Responsibilities (worker + main process):
  1. Register the target ops from ops/<op>/register.py with explicit platform.
  2. Patch vllm_fl WorkerFL.execute_model to snapshot CALL_COUNT before/after
     each forward call -> proves *generation-phase* consumption (not init-only).
  3. At process exit, dump one JSON per PID with run_id/role/pid/ppid,
     init_count, per-execute deltas and final_count.

Env:
  FLAGOS_C2_RUN_ID     run id (enables this hook)
  FLAGOS_A1_OPS        comma-separated op names (empty => baseline, no reg)
  A1_PLATFORM          ascend (default)
  A1_DISPATCH_KEY      AutogradPrivateUse1 (default)
  FLAGOS_C2_COUNT_DIR  directory for per-PID dump files
  FLAGOS_TEMPLATES_ROOT repo root (defaults to this file's ../..)
"""
import atexit
import importlib
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_RUN_ID = os.environ.get("FLAGOS_C2_RUN_ID")

if _RUN_ID:
    root = os.environ.get("FLAGOS_TEMPLATES_ROOT", _ROOT)
    if root and root not in sys.path:
        sys.path.insert(0, root)

    _ops = [x for x in os.environ.get("FLAGOS_A1_OPS", "").split(",") if x]
    _key = os.environ.get("A1_DISPATCH_KEY", "AutogradPrivateUse1")
    _plat = os.environ.get("A1_PLATFORM", "ascend")
    _dump_dir = os.environ.get("FLAGOS_C2_COUNT_DIR", "")

    _regmods: dict = {}
    for _op in _ops:
        try:
            pkg = __import__(f"ops.{_op}", fromlist=["register_a1"])
            pkg.register_a1(_key, platform=_plat)
            _regmods[_op] = importlib.import_module(f"ops.{_op}.register")
        except Exception as e:  # noqa: BLE001
            print(f"[c2-hook] register failed ops.{_op}: {e!r}", file=sys.stderr)

    def _counts():
        return {op: m.CALL_COUNT.get(op, 0) for op, m in _regmods.items()}

    _exec_calls = []  # [{"before": {...}, "after": {...}}]
    _patched = False
    if _regmods:
        try:
            import vllm_fl.worker.worker as _w

            _cls = _w.WorkerFL
            _orig = _cls.execute_model

            def execute_model(self, *args, **kwargs):
                before = _counts()
                out = _orig(self, *args, **kwargs)
                after = _counts()
                _exec_calls.append({"before": before, "after": after})
                return out

            _cls.execute_model = execute_model
            _patched = True
        except Exception as e:  # noqa: BLE001
            print(f"[c2-hook] execute_model patch failed: {e!r}",
                  file=sys.stderr)

    def _dump():
        if not _dump_dir:
            return
        try:
            os.makedirs(_dump_dir, exist_ok=True)
            final = _counts()
            init = {op: 0 for op in _regmods}
            gen = {op: 0 for op in _regmods}
            if _exec_calls:
                first = _exec_calls[0]["before"]
                init = dict(first)
                for call in _exec_calls:
                    for op in _regmods:
                        gen[op] += call["after"][op] - call["before"][op]
            else:
                init = dict(final)
            rec = {
                "run_id": _RUN_ID,
                "role": os.environ.get("VLLM_WORKER_MULTIPROC_ROLE", "unknown"),
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "patched": _patched,
                "ops": _ops,
                "platform": _plat,
                "dispatch_key": _key,
                "init_count": init,
                "generation_hits": gen,
                "final_count": final,
                "execute_calls": len(_exec_calls),
            }
            p = os.path.join(_dump_dir, f"count_{os.getpid()}.json")
            with open(p, "w") as f:
                json.dump(rec, f)
        except Exception:  # noqa: BLE001
            pass

    if _dump_dir:
        atexit.register(_dump)
