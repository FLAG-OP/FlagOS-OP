#!/usr/bin/env python3
"""AI 生成算子 intake 验证: manifest 契约 → kernel/op/framework 三级。

用法:
  python3 scripts/intake_validate.py intake/cases/kernelgen-gelu-example
  python3 scripts/intake_validate.py                        # 全部 intake case
  python3 scripts/intake_validate.py --validate-only        # 仅契约校验（CI）

层级:
  kernel 层  精度 vs 参考 · 哨兵（确定性+输入敏感）· 性能（入 perf 记录）
  框架层     A1 aten 注册 或 A2 dispatch 注册 + 钉选调用
  应用层     experimental，默认 SKIP（需 vLLM 引擎，见 docs/ai-intake.md）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INTAKE_DIR = ROOT / "intake" / "cases"
OUT_DIR = ROOT / "results" / "intake"

DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


# ============ 契约校验（不依赖 torch，CI 可直接跑） ============

def validate_manifest(m: dict, case_dir: Path) -> list[str]:
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)

    need(isinstance(m, dict), "manifest 必须是 JSON 对象")
    if not isinstance(m, dict):
        return errs
    for k in ("schema", "name", "source", "op", "kernel", "inputs",
              "tolerance", "targets"):
        need(k in m, f"缺少必填字段: {k}")
    if errs:
        return errs

    need(m["schema"] == 1, "schema 必须为 1")
    need(m["source"] in ("kernelgen", "kernelbench", "manual"),
         "source 必须是 kernelgen/kernelbench/manual")
    need(m.get("expect", "ok") in ("ok", "blocked"),
         "expect 必须是 ok/blocked")

    for sec in ("kernel", "reference"):
        if sec in m:
            e = m[sec]
            need(isinstance(e, dict) and "module" in e and "entry" in e,
                 f"{sec} 需要 module/entry")
            if isinstance(e, dict) and "module" in e:
                need((case_dir / e["module"]).is_file(),
                     f"{sec}.module 不存在: {e['module']}")

    need("reference" in m or m["op"] in ("gelu_and_mul", "silu", "relu"),
         "无 reference 时 op 必须是内置语义参考(gelu_and_mul/silu/relu)")

    ins = m["inputs"]
    need(isinstance(ins, list) and len(ins) >= 1, "inputs 必须是非空数组")
    for i, x in enumerate(ins if isinstance(ins, list) else []):
        need(isinstance(x, dict) and "shape" in x and "dtype" in x,
             f"inputs[{i}] 缺 shape/dtype")
        if isinstance(x, dict):
            need(x.get("dtype") in DTYPES, f"inputs[{i}].dtype 非法")
            sh = x.get("shape")
            need(isinstance(sh, list) and all(
                 isinstance(d, int) and d > 0 for d in sh),
                 f"inputs[{i}].shape 非法")

    tol = m["tolerance"]
    need(isinstance(tol, dict) and isinstance(tol.get("max_abs_err"), (int, float))
         and tol["max_abs_err"] > 0, "tolerance.max_abs_err 必须为正数")

    t = m["targets"]
    need(isinstance(t, dict), "targets 必须是对象")
    if isinstance(t, dict):
        need(isinstance(t.get("routes"), list) and
             set(t["routes"]) <= {"a1", "a2"} and t["routes"],
             "targets.routes 必须是 a1/a2 的非空子集")
        need(isinstance(t.get("levels"), list) and
             set(t["levels"]) <= {"kernel", "op", "framework"} and t["levels"],
             "targets.levels 必须是 kernel/op/framework 的非空子集")
        if "op" in t.get("levels", []):
            need("framework" in m and (m["framework"].get("a1")
                                       or m["framework"].get("a2")),
                 "op 层验证需要 framework.a1 或 framework.a2 注册信息")
    return errs


# ============ 加载 ============

def load_entry(case_dir: Path, section: dict):
    path = case_dir / section["module"]
    name = "intake_" + path.stem + "_" + section["entry"]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, section["entry"])


def make_inputs(m: dict, device: str, shape_override=None, dtype_override=None):
    import torch
    parts = []
    for i, x in enumerate(m["inputs"]):
        shape = tuple(shape_override or x["shape"])
        dt = getattr(torch, DTYPES[dtype_override or x["dtype"]])
        seed = x.get("seed", 20260827 + i)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t = torch.randn(*shape, dtype=torch.float32, generator=gen).to(dt)
        t = t * x.get("scale", 1.0)
        parts.append(t.to(device))
    return parts


def load_reference(m: dict, case_dir: Path):
    if "reference" in m:
        return load_entry(case_dir, m["reference"])
    from common.kernel_spec import SEMANTIC_REFS
    return SEMANTIC_REFS[m["op"]]


# ============ 三级验证 ============

def _r(status, detail, **extra):
    return {"status": status, "detail": detail, **extra}


def run_kernel_level(m: dict, case_dir: Path, profile) -> dict:
    import torch
    from common.perf import PerfCase, run_case, save_run_record

    dev = profile.torch_device
    fn = load_entry(case_dir, m["kernel"])
    ref = load_reference(m, case_dir)
    parts = make_inputs(m, dev)

    # 精度
    out = fn(*parts)
    ref_out = ref(*[p.clone() for p in parts])
    if out.shape != ref_out.shape:
        return _r("FAIL", f"shape 不匹配 kernel={tuple(out.shape)} "
                          f"ref={tuple(ref_out.shape)}")
    err = (out.float() - ref_out.float()).abs().max().item()
    if err > m["tolerance"]["max_abs_err"]:
        return _r("FAIL", f"精度 err={err:.4f} > "
                          f"{m['tolerance']['max_abs_err']}", max_err=err)

    # 哨兵: 确定性 + 输入敏感（拦截 known-issues #2/#11 类静默缺陷）
    o1, o2 = fn(*[p.clone() for p in parts]), fn(*[p.clone() for p in parts])
    torch.cuda.synchronize()
    det = torch.equal(o1, o2)
    perturbed = [p + 1.0 for p in parts]
    o3 = fn(*perturbed)
    torch.cuda.synchronize()
    sens = not torch.equal(o1, o3)
    if not (det and sens):
        return _r("FAIL", f"哨兵: 确定性={'✓' if det else '✗'} "
                          f"输入敏感={'✓' if sens else '✗'}",
                  deterministic=det, sensitive=sens)

    # 性能（perf 段可覆盖 shape/dtype/采样参数）
    perf = m.get("perf", {})
    shape = tuple(perf.get("shape", parts[0].shape))
    dtype = perf.get("dtype", m["inputs"][0]["dtype"])
    pparts = make_inputs(m, dev, shape_override=list(shape),
                         dtype_override=dtype)
    nbytes = sum(p.numel() * p.element_size() for p in pparts)
    out_numel = pparts[0].numel() if len(pparts) == 1 else pparts[0].numel()

    case = PerfCase(
        case_id=f"intake.{m['name']}", group="intake", level="kernel",
        make_fn=lambda p: (lambda: fn(*pparts)),
        derived=lambda t: {"GBps": (nbytes + out_numel
                                    * pparts[0].element_size()) / t / 1e6},
        warmup=perf.get("warmup", 20), iters=perf.get("iters", 100))
    rec = run_case(case, profile)
    save_run_record(rec)
    return _r("PASS", f"精度 err={err:.2e} · 哨兵 OK · "
                      f"性能 {rec.metrics['latency_ms']:.3f}ms "
                      f"({rec.metrics.get('GBps', 0):.0f} GB/s)",
              max_err=err, perf=rec.metrics)


def run_op_level(m: dict, case_dir: Path, profile) -> dict:
    fn = load_entry(case_dir, m["kernel"])
    fw = m.get("framework", {})
    parts = make_inputs(m, profile.torch_device)

    if fw.get("a2"):
        try:
            from vllm_fl.dispatch import (call_op, get_default_manager,
                                          reset_default_manager)
            from vllm_fl.dispatch.backends.base import Backend
            from vllm_fl.dispatch.policy import (with_allowed_vendors,
                                                 with_preference)
            from vllm_fl.dispatch.types import (BackendImplKind,
                                                BackendPriority, OpImpl)
        except ImportError as e:
            return _r("SKIP", f"vllm_fl 不可用: {e}")

        cfg = fw["a2"]

        class IntakeBackend(Backend):
            @property
            def name(self):
                return cfg["vendor"]

            @property
            def vendor(self):
                return cfg["vendor"]

            def is_available(self):
                return True

            def op(self, obj, *args):
                return fn(*args)

        try:
            reset_default_manager()
            mgr = get_default_manager()
            mgr.ensure_initialized()
            mgr.registry.register_many([OpImpl(
                op_name=cfg["op_name"], impl_id=cfg["impl_id"],
                kind=BackendImplKind.VENDOR, fn=IntakeBackend().op,
                vendor=cfg["vendor"], priority=BackendPriority.VENDOR)])
            with with_preference("vendor"), with_allowed_vendors(cfg["vendor"]):
                out = call_op(cfg["op_name"], None, *parts)
                used = mgr._called_ops.get(cfg["op_name"])
            if used != cfg["impl_id"]:
                return _r("FAIL", f"dispatch 选择 {used} != {cfg['impl_id']}")
            ref = load_reference(m, case_dir)
            ref_out = ref(*[p.clone() for p in parts])
            import torch
            err = (out.float() - ref_out.float()).abs().max().item()
            if err > m["tolerance"]["max_abs_err"]:
                return _r("FAIL", f"注册后精度 err={err:.4f}")
            return _r("PASS", f"A2 注册 {cfg['impl_id']} · 钉选调用 · err={err:.2e}")
        except Exception:
            return _r("FAIL", traceback.format_exc(limit=2).splitlines()[-1])

    if fw.get("a1"):
        import torch
        aten_op = fw["a1"]["aten_op"]
        calls = {"n": 0}

        def wrapper(*args, **kw):
            calls["n"] += 1
            return fn(*args, **kw)

        try:
            lib = torch.library.Library("aten", "IMPL")
            lib.impl(aten_op, wrapper, profile.dispatch_key)
            if hasattr(torch, aten_op):
                getattr(torch, aten_op)(*parts)
            else:
                torch.ops.aten[aten_op](*parts)
            if calls["n"] == 0:
                return _r("FAIL", f"aten::{aten_op} 未被拦截")
            return _r("PASS", f"A1 注册 aten::{aten_op} · 拦截 {calls['n']} 次")
        except Exception:
            return _r("FAIL", traceback.format_exc(limit=2).splitlines()[-1])

    return _r("SKIP", "manifest 未声明框架层注册信息")


def run_framework_level(m: dict) -> dict:
    if "framework" not in m["targets"]["levels"]:
        return _r("SKIP", "manifest 未申请应用层验证")
    return _r("SKIP", "experimental: 需 vLLM 引擎，参照 b-fullstack 的 "
                      "PER_OP 注入方式接入（docs/ai-intake.md）")


# ============ 主流程 ============

def validate_case(case_dir: Path, profile) -> tuple[dict, int]:
    m = json.loads((case_dir / "manifest.json").read_text())
    errs = validate_manifest(m, case_dir)
    if errs:
        return {"name": case_dir.name, "manifest": "FAIL", "errors": errs,
                "levels": {}}, 1

    levels = {}
    levels["kernel"] = run_kernel_level(m, case_dir, profile)
    if "op" in m["targets"]["levels"]:
        levels["op"] = run_op_level(m, case_dir, profile)
    if "framework" in m["targets"]["levels"]:
        levels["framework"] = run_framework_level(m)

    expect = m.get("expect", "ok")
    kern_ok = levels["kernel"]["status"] == "PASS"
    if expect == "ok":
        failed = [k for k, v in levels.items() if v["status"] == "FAIL"]
        ok = not failed and kern_ok
        overall = "PROMOTED" if ok else "BLOCKED"
        code = 0 if ok else 1
    else:  # 负例: kernel 层应被拦截
        ok = levels["kernel"]["status"] == "FAIL"
        overall = "BLOCKED-OK（负例检出）" if ok else "NOT-DETECTED"
        code = 0 if ok else 1

    result = {"name": m["name"], "source": m["source"], "expect": expect,
              "manifest": "PASS", "levels": levels, "overall": overall,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    return result, code


def write_report(result: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{result['name']}.json").write_text(json.dumps(result, indent=2))
    rows = "\n".join(
        f"| {k} | {v['status']} | {v['detail']} |" for k, v in result["levels"].items())
    md = [
        f"# Intake 报告: {result['name']}",
        "",
        f"- 结论: **{result['overall']}** · source={result['source']} "
        f"· expect={result['expect']}",
        f"- 时间: {result['timestamp']}",
        "",
        "| 层级 | 结果 | 说明 |", "|---|---|---|", rows or "| — | — | — |",
        "",
    ]
    p = OUT_DIR / f"{result['name']}_report.md"
    p.write_text("\n".join(md))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", nargs="?", default=None,
                    help="case 目录（缺省全部 intake/cases/*）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--validate-only", action="store_true",
                    help="仅校验 manifest 契约（不导入 torch）")
    args = ap.parse_args()

    dirs = ([Path(args.case)] if args.case
            else sorted(p for p in INTAKE_DIR.iterdir() if p.is_dir()))
    if not dirs or not all((d / "manifest.json").is_file() for d in dirs):
        print("未找到含 manifest.json 的 intake case")
        return 2

    if args.validate_only:
        n_err = 0
        for d in dirs:
            m = json.loads((d / "manifest.json").read_text())
            errs = validate_manifest(m, d)
            print(f"  {d.name:40s} {'OK' if not errs else 'FAIL'}")
            for e in errs:
                print(f"    - {e}")
            n_err += len(errs)
        print(f"契约校验: {len(dirs)} case · {n_err} 错误")
        return 1 if n_err else 0

    from common.device import detect_profile, load_profile
    profile = load_profile(args.device) if args.device else detect_profile()
    print(f"[intake] 设备: {profile.summary()}")

    code = 0
    for d in dirs:
        print("-" * 60)
        result, c = validate_case(d, profile)
        code |= c
        p = write_report(result)
        print(f"  {result['name']:40s} {result['overall']}")
        for k, v in result.get("levels", {}).items():
            print(f"    {k:10s} {v['status']:5s} {v['detail'][:90]}")
        for e in result.get("errors", []):
            print(f"    manifest   FAIL  {e}")
        print(f"  报告: {p}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
