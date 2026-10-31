#!/usr/bin/env python3
"""C2 真实 vLLM-Ascend 双跑 harness——embedding（issue #38，阶段归因版）。

与 sdpa_math/script/c2_real_vllm_ascend.py 同构（阶段/断言/性能口径一致），
注入目标为 aten::embedding（A1 拦截，dispatch key PrivateUse1）——注意
embedding 无注意力路径差异，inject 与 baseline 的调用链完全同构，唯一
变量 = 是否注册 FlagOS A1 embedding 实现。

断言（预注册于本文件，依据 docs/testing.md#assertions，不得事后放宽）:
  C1 恒等（首选）: baseline vs inject 逐 prompt 生成前缀完全一致（6/6 位级）。
  C2 下限: 前 2 token 一致率 ≥ 2/3 × 6。
  INCONCLUSIVE/FAIL 处置同 sdpa（补首分歧/中间张量/logits 排查）。

阶段计数（意见 3）: FLAGOS_C2_RUN_ID 贯穿 + hit_times 时间轴把引擎/主进程
命中拆到 init/correctness/warmup/measured；旧文件剔除；pid/ppid 角色判定。

性能（意见 4）: 预热 + K 次 generate 计时，同路径门禁 = inject/baseline
中位数比 ≤1.2 OK、≤1.3 WARN、>1.3 FAIL（对齐 docs/performance-regression.md）。

用法:
  python3 ops/embedding/script/c2_real_vllm_ascend.py --out reports/c2_ascend910
  # 默认 --mode accept: 命中/正确性/性能门禁/归因任一未过 → 非零退出
  # --mode collect: 显式采集模式，采集完成即 0（不得被当作验收通过）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from c2_acceptance import decide  # noqa: E402  纯函数验收判定（意见 4）
MODEL_DEFAULT = (
    "/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
    "/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
)

# 预注册 prompt 集（与 sdpa_math harness 完全相同，便于跨算子对照）。
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10)) #",
    "Once upon a time, in a land far away, there lived a",
    "Water boils at 100 degrees",
    "2 + 2 = 4, 3 + 3 =",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]
PROMPT = PROMPTS[0]
MAX_TOKENS = 8
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 5
IDENTITY_ALL_PROMPTS = True
CUSTOM_MIN_FIRST2_RATE = 2.0 / 3.0
PERF_OK = 1.2
PERF_WARN = 1.3
SYNC_NOTE = (
    "llm.generate 为同步阻塞调用；模型前向在 EngineCore 子进程执行，"
    "主进程无法对引擎进程调用 torch.npu.synchronize()，故计时为 "
    "generate 墙钟时延（含调度/排队，非纯 kernel 时延）；两组同步方式"
    "完全一致，且每组独立进程运行。注入组计数落盘开销见 "
    "hit_files[].dump_s（相对 generate 时延为微秒级）。"
)
_CANN = [
    "/usr/local/Ascend/cann-9.0.0/python/site-packages",
    "/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe",
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
]
MODES = ("baseline", "inject")


def _child_env(mode: str, workdir: pathlib.Path, run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(ROOT / "injection"), "/vllm-workspace/vllm-ascend", *_CANN]
    env["PYTHONPATH"] = ":".join(parts + [p for p in [env.get("PYTHONPATH")] if p])
    env["VLLM_PLUGINS"] = "ascend"
    env["ASCEND_HOME_PATH"] = "/usr/local/Ascend/cann-9.0.0"
    env["HCCL_VERIFY"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["FLAGOS_C2_RUN_ID"] = run_id
    for k in ("FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS", "FLAGOS_C2_COUNT_DIR",
              "A1_DISPATCH_KEY", "FLAGOS_SDPA_MATH_ATTN", "FLAGOS_A1_IMPL"):
        env.pop(k, None)
    if mode == "inject":
        env["FLAGOS_A1_ATEN_INJECT"] = "1"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["FLAGOS_A1_OPS"] = "embedding"
        # 推理（no_grad）路径 dispatch 到 PrivateUse1；
        # AutogradPrivateUse1 已被 vllm_ascend 注册，会冲突。
        env["A1_DISPATCH_KEY"] = "PrivateUse1"
        env["FLAGOS_C2_COUNT_DIR"] = str(workdir)
    return env


_CHILD_RUNNER = r"""
import json, math, os, pathlib, statistics, sys, time, traceback

mode, model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.loads(os.environ["FLAGOS_C2_PROMPTS_JSON"])
max_tokens = int(os.environ.get("FLAGOS_C2_MAX_TOKENS", "8"))
warmup = int(os.environ.get("FLAGOS_C2_WARMUP", "1"))
repeats = int(os.environ.get("FLAGOS_C2_REPEATS", "5"))
run_id = os.environ.get("FLAGOS_C2_RUN_ID")
result = {"mode": mode, "ok": False, "pid": os.getpid(), "ppid": os.getppid(),
          "run_id": run_id}
phases = {}
main_snaps = {}
counter = None


def _snap(name):
    if counter is not None:
        main_snaps[name] = int(counter.get("n", 0))


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        phases[self.name] = {"start": round(self.t0, 6),
                             "end": round(time.time(), 6)}


try:
    with _Phase("setup"):
        import torch
        import torch_npu  # noqa: F401
        result["torch"] = torch.__version__
        result["torch_npu"] = getattr(__import__("torch_npu"), "__version__", "?")
        result["npu_available"] = bool(hasattr(torch, "npu") and torch.npu.is_available())
        result["npu_count"] = torch.npu.device_count() if result["npu_available"] else 0
        result["mp_method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
        result["a1_inject"] = os.environ.get("FLAGOS_A1_ATEN_INJECT", "0")
        result["a1_impl_env"] = os.environ.get("FLAGOS_A1_IMPL")
        if mode == "inject":
            root = pathlib.Path(os.environ["FLAGOS_TEMPLATES_ROOT"])
            sys.path.insert(0, str(root / "injection"))
            from a1_aten_ops import get_counter, get_reg_info
            counter = get_counter("embedding")
            result["a1_counter_retrieved"] = counter is not None
            # 意见 5: 注册参数事实（embedding register_a1 不接受 impl，
            # 旧实现设 FLAGOS_A1_IMPL 时会经 TypeError 回退把 counter
            # 一起丢掉；新实现按签名分发并如实记录 impl_applied=false）
            result["a1_reg_info"] = get_reg_info("embedding")

    from vllm import LLM, SamplingParams
    import vllm
    result["vllm"] = getattr(vllm, "__version__", "?")
    try:
        import vllm_ascend
        result["vllm_ascend"] = str(getattr(vllm_ascend, "__version__", "source"))
        result["vllm_ascend_file"] = getattr(vllm_ascend, "__file__", None)
    except Exception as e:
        result["vllm_ascend_err"] = repr(e)

    with _Phase("init"):
        t0 = time.time()
        llm = LLM(model=model, enforce_eager=True, max_model_len=192,
                  gpu_memory_utilization=0.2, max_num_seqs=1,
                  trust_remote_code=True, disable_log_stats=True)
        result["init_s"] = round(time.time() - t0, 2)
    _snap("after_init")

    with _Phase("correctness"):
        t1 = time.time()
        outs = None
        try:
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                prompt_logprobs=0)
            outs = llm.generate(prompts, sp)
        except Exception:
            result["prompt_logprobs_err"] = traceback.format_exc().splitlines()[-1]
            sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
            outs = llm.generate(prompts, sp)
        result["correctness_s"] = round(time.time() - t1, 2)
        result["outputs"] = []
        result["prompt_logprobs"] = {}
        for idx, o in enumerate(outs):
            result["outputs"].append({
                "prompt_idx": idx,
                "prompt": prompts[idx],
                "token_ids": list(o.outputs[0].token_ids[:max_tokens]),
                "text": o.outputs[0].text,
            })
            plmap = {}
            for pos, d in enumerate(o.prompt_logprobs or []):
                if not d:
                    continue
                lp = next(iter(d.values()))
                plmap[str(pos)] = float(getattr(lp, "logprob", lp))
            if plmap:
                result["prompt_logprobs"][str(idx)] = plmap
    _snap("after_correctness")

    sp_perf = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with _Phase("warmup"):
        for _ in range(max(warmup, 0)):
            llm.generate([prompts[0]], sp_perf)
    _snap("after_warmup")
    samples = []
    with _Phase("measured"):
        for _ in range(max(repeats, 1)):
            tp = time.perf_counter()
            llm.generate([prompts[0]], sp_perf)
            samples.append(time.perf_counter() - tp)
    _snap("after_measured")
    if samples:
        _s = sorted(samples)
        _p95_idx = max(0, math.ceil(0.95 * len(_s)) - 1)
        result["perf"] = {
            "warmup": warmup,
            "repeats": len(samples),
            "samples_s": [round(x, 4) for x in samples],
            "stats_s": {
                "mean": round(statistics.fmean(samples), 4),
                "median": round(statistics.median(samples), 4),
                "p95": round(_s[_p95_idx], 4),
                "min": round(min(samples), 4),
                "max": round(max(samples), 4),
                "stdev": round(statistics.pstdev(samples), 4)
                         if len(samples) > 1 else 0.0,
            },
        }
    result["ok"] = True
except Exception:
    result["error"] = traceback.format_exc()

result["phases"] = phases
result["main_counter_snapshots"] = main_snaps
result["main_counter_scope"] = (
    "仅本进程（child runner）A1 计数快照；引擎子进程计数须读 "
    "FLAGOS_C2_COUNT_DIR 下的 *_hits_<pid>.json 并按 hit_times 归阶段")
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
"""


def run_mode(mode: str, model: str, out_dir: pathlib.Path, run_id: str,
             warmup: int, repeats: int, timeout_s: int = 900) -> dict:
    out_path = out_dir / f"{mode}.json"
    for old in out_dir.glob("embedding_hits_*.json"):
        old.unlink()
    env = _child_env(mode, workdir=out_dir, run_id=run_id)
    env["FLAGOS_C2_PROMPTS_JSON"] = json.dumps(PROMPTS, ensure_ascii=False)
    env["FLAGOS_C2_MAX_TOKENS"] = str(MAX_TOKENS)
    env["FLAGOS_C2_WARMUP"] = str(warmup)
    env["FLAGOS_C2_REPEATS"] = str(repeats)
    cmd = [sys.executable, "-c", _CHILD_RUNNER, mode, model, str(out_path)]
    print(f"[{mode}] start run_id={run_id}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), timeout=timeout_s,
                          capture_output=True, text=True)
    elapsed = round(time.time() - t0, 2)
    (out_dir / f"{mode}.stdout.log").write_text(proc.stdout or "")
    (out_dir / f"{mode}.stderr.log").write_text(proc.stderr or "")
    data: dict = {}
    if out_path.is_file():
        data = json.loads(out_path.read_text())
    data.setdefault("mode", mode)
    data["rc"] = proc.returncode
    data["wall_s"] = elapsed
    data["ok"] = bool(data.get("ok")) and proc.returncode == 0
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[{mode}] rc={proc.returncode} ok={data.get('ok')} wall={elapsed}s",
          flush=True)
    return data


def merge_hits(out_dir: pathlib.Path, run_id: str) -> tuple[list[dict], list[dict]]:
    keep, stale = [], []
    for p in sorted(out_dir.glob("embedding_hits_*.json")):
        try:
            h = json.loads(p.read_text())
        except Exception:
            stale.append({"path": str(p), "reason": "unreadable"})
            continue
        h["path"] = str(p)
        if h.get("run_id") != run_id:
            h["reason"] = f"stale run_id={h.get('run_id')!r}"
            stale.append(h)
            continue
        keep.append(h)
    return keep, stale


def attribute_hits(hits: list[dict], inject: dict, phases: dict) -> dict:
    main_pid = inject.get("pid")
    role_split = {"main": {}, "engine/worker": {}, "unknown": {}}
    outside = 0
    for h in hits:
        pid = h.get("pid")
        if main_pid is None or pid is None:
            h["role"] = "unknown"
        elif pid == main_pid:
            h["role"] = "main"
        else:
            h["role"] = "engine/worker"
        h["ppid_match_main"] = (h.get("ppid") == main_pid)
        buckets = role_split.setdefault(h["role"], {})
        buckets["hits"] = buckets.get("hits", 0) + int(h.get("hits") or 0)
        phase_counts: dict[str, int] = {}
        windows = [(name, w["start"], w["end"])
                   for name, w in (phases or {}).items()]
        for t in (h.get("hit_times") or []):
            for name, s, e in windows:
                if s <= t <= e:
                    phase_counts[name] = phase_counts.get(name, 0) + 1
                    break
            else:
                phase_counts["outside_windows"] = phase_counts.get(
                    "outside_windows", 0) + 1
        h["phase_counts"] = phase_counts
        for k, v in phase_counts.items():
            buckets[k] = buckets.get(k, 0) + v
            if k == "outside_windows":
                outside += v
    return {"by_role": role_split, "outside_windows": outside}


def _prefix(a: list, b: list) -> tuple[int, int | None]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            return n, n
        n += 1
    if len(a) != len(b):
        return n, n
    return n, None


def correctness_compare(base: dict, inject: dict) -> dict:
    if not (base.get("ok") and inject.get("ok")):
        return {"applicable": False, "reason": "base/inject 未全部成功"}
    b_outs = {o["prompt_idx"]: o for o in base.get("outputs") or []}
    i_outs = {o["prompt_idx"]: o for o in inject.get("outputs") or []}
    per_prompt, n_identity, n_first2 = [], 0, 0
    for idx in range(len(PROMPTS)):
        b = list((b_outs.get(idx) or {}).get("token_ids") or [])
        i = list((i_outs.get(idx) or {}).get("token_ids") or [])
        pre, div = _prefix(b, i)
        first2_ok = b[:2] == i[:2]
        bitwise = b == i
        n_identity += int(bitwise)
        n_first2 += int(first2_ok)
        per_prompt.append({
            "prompt_idx": idx,
            "prompt": (b_outs.get(idx) or {}).get("prompt")
                      or (i_outs.get(idx) or {}).get("prompt"),
            "baseline_token_ids": b,
            "inject_token_ids": i,
            "common_prefix_len": pre,
            "first_divergence_token_index": div,
            "bitwise_equal": bitwise,
            "first2_equal": first2_ok,
        })
    n = len(PROMPTS)
    rate = n_first2 / n if n else 0.0
    c1 = (n_identity == n) if IDENTITY_ALL_PROMPTS else None
    c2 = rate >= CUSTOM_MIN_FIRST2_RATE
    verdict = "PASS" if c1 else ("INCONCLUSIVE" if c2 else "FAIL")

    def _lp(b: dict, i: dict) -> dict:
        bd, idd = b.get("prompt_logprobs") or {}, i.get("prompt_logprobs") or {}
        diffs, tokens = [], 0
        per_prompt_out = {}
        for k in sorted(set(bd) & set(idd), key=int):
            common = sorted(set(bd[k]) & set(idd[k]), key=int)
            if not common:
                continue
            ds = [abs(float(bd[k][p]) - float(idd[k][p])) for p in common]
            diffs.extend(ds)
            tokens += len(common)
            per_prompt_out[k] = {"tokens": len(common),
                                 "max_abs_diff": round(max(ds), 6),
                                 "mean_abs_diff": round(sum(ds) / len(ds), 6)}
        return {"prompts_compared": len(per_prompt_out), "tokens": tokens,
                "max_abs_diff": round(max(diffs), 6) if diffs else None,
                "mean_abs_diff": round(sum(diffs) / len(diffs), 6) if diffs else None,
                "per_prompt": per_prompt_out}

    return {
        "applicable": True,
        "criterion_predeclared": {
            "C1_identity": "baseline vs inject 逐 prompt 生成前缀完全一致"
                            "（6/6 prompt，位级）",
            "C2_custom_min": f"前 2 token 一致率 ≥ {CUSTOM_MIN_FIRST2_RATE:.3f}"
                              f"（≥{int(CUSTOM_MIN_FIRST2_RATE * n)}/{n} prompt）",
            "policy": "未过 C1 但过 C2 → INCONCLUSIVE（补首分歧/中间张量/"
                      "logits 排查）；两皆未过 → FAIL；不得事后放宽阈值"
                      "（docs/testing.md#assertions）",
        },
        "per_prompt": per_prompt,
        "identity_pass": c1,
        "identity_pass_count": f"{n_identity}/{n}",
        "first2_match_rate": round(rate, 4),
        "first2_match": f"{n_first2}/{n}",
        "custom_min_pass": c2,
        "correctness_verdict": verdict,
        "prompt_logprob_same_path": _lp(base, inject),
    }


def perf_compare(perfs: dict[str, dict | None]) -> dict:
    def med(mode: str):
        p = perfs.get(mode) or {}
        return ((p.get("stats_s") or {}).get("median"))

    comparisons = [
        {"name": "inject_vs_baseline", "gate": True,
         "a": "inject", "b": "baseline"},
    ]
    out = []
    for c in comparisons:
        a, b = med(c["a"]), med(c["b"])
        if not a or not b:
            out.append({**c, "available": False})
            continue
        ratio = round(a / b, 4)
        verdict = ("OK" if ratio <= PERF_OK
                   else "WARN" if ratio <= PERF_WARN else "FAIL")
        out.append({**c, "available": True, "a_median_s": a, "b_median_s": b,
                    "ratio_median": ratio, "verdict": verdict})
    gate = next((x for x in out if x["gate"]), {})
    return {
        "thresholds_predeclared": {"ok": PERF_OK, "warn": PERF_WARN,
                                   "note": "同调用链 inject/baseline 中位数比；"
                                           "对齐 docs/performance-regression.md "
                                           "+20%/+30%"},
        "sync_note": SYNC_NOTE,
        "per_mode": {m: p for m, p in perfs.items() if p},
        "comparisons": out,
        "gate_verdict": gate.get("verdict", "MISSING"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FLAGOS_MODEL_PATH",
                                                      MODEL_DEFAULT))
    ap.add_argument("--out", default=str(
        pathlib.Path(__file__).resolve().parent.parent / "reports" / "c2_ascend910"))
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--mode", choices=("accept", "collect"), default="accept",
                    help="accept: 全部验收必需项通过才 exit 0（CI/验收入口）；"
                         "collect: 显式采集模式，采集完成 exit 0（非验收）")
    args = ap.parse_args()
    model = args.model
    if not pathlib.Path(model).is_dir():
        print(f"model path missing: {model}", file=sys.stderr)
        return 2
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    run_id = os.environ.get("FLAGOS_C2_RUN_ID") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{os.getpid()}")
    (out_dir / "run_id.txt").write_text(run_id + "\n")

    results: dict[str, dict] = {}
    for mode in MODES:
        if args.skip_baseline and mode != "inject":
            results[mode] = {"mode": mode, "ok": False, "skipped": True}
            continue
        results[mode] = run_mode(mode, model, out_dir, run_id,
                                 args.warmup, args.repeats)
    hits, stale = merge_hits(out_dir, run_id)
    inject = results["inject"]
    phases = inject.get("phases") or {}
    for h in hits:
        if inject.get("pid") is None or h.get("pid") is None:
            h["role"] = "unknown"
        else:
            h["role"] = "main" if h["pid"] == inject["pid"] else "engine/worker"
    attr = attribute_hits(hits, inject, phases)
    engine_hits = int(attr["by_role"].get("engine/worker", {}).get("hits") or 0)
    main_hits = int(attr["by_role"].get("main", {}).get("hits") or 0)
    forward_hits = engine_hits
    if (forward_hits or 0) == 0:
        forward_hits = inject.get("main_counter_snapshots", {}).get(
            "after_correctness") or 0
    hits_total = sum(int(h.get("hits") or 0) for h in hits)
    phase_total = sum(
        int(v or 0) for role, bucket in attr["by_role"].items()
        for k, v in bucket.items() if k != "hits")

    baseline, inj = results["baseline"], results["inject"]
    correctness = correctness_compare(baseline, inj)
    perfs = {m: results[m].get("perf") for m in MODES}
    perf = perf_compare(perfs)

    acceptance = {
        "hit": bool((forward_hits or 0) > 0),
        "correctness_same_path": correctness.get("correctness_verdict"),
        "perf_gate": perf["gate_verdict"],
        "rc_all_zero": all(results[m].get("rc") == 0 for m in MODES
                           if not results[m].get("skipped")),
    }
    complete = (acceptance["hit"]
                and acceptance["correctness_same_path"] == "PASS"
                and acceptance["perf_gate"] == "OK")
    acceptance["status"] = ("complete-pending-review" if complete else "PARTIAL")

    summary = {
        "schema": 3,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": "ascend910",
        "engine": "vllm-ascend",
        "commit": commit,
        "run_id": run_id,
        "model": model,
        "prompts": PROMPTS,
        "max_tokens": MAX_TOKENS,
        "python": platform.python_version(),
        "injection": {
            "mp_method": inject.get("mp_method"),
            "a1_ops": "embedding",
            "a1_dispatch_key": "PrivateUse1",
            # 实际值取 inject 子进程亲测 env（意见 5: 显式清理/如实落盘；
            # embedding 路径不设置 FLAGOS_A1_IMPL，故为 None）
            "a1_impl": inject.get("a1_impl_env"),
            "a1_reg_info": inject.get("a1_reg_info"),
        },
        "controls": {
            "note": "两组共用同一模型/权重快照/prompt 集/采样参数(温度0)"
                    "/dtype/enforce_eager/engine 配置；调用链完全同构，"
                    "唯一变量 = 是否注册 FlagOS A1 embedding 实现",
            "modes": list(MODES),
            "warmup": args.warmup, "repeats": args.repeats,
            "env_overrides_inject": [
                "FLAGOS_A1_ATEN_INJECT=1",
                "FLAGOS_A1_OPS=embedding",
                "A1_DISPATCH_KEY=PrivateUse1",
                "FLAGOS_C2_RUN_ID=<run_id>",
                "FLAGOS_A1_IMPL=<unset — embedding register_a1 不接受 impl>",
            ],
            "env_cleared_all_modes": [
                "FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS", "FLAGOS_C2_COUNT_DIR",
                "A1_DISPATCH_KEY", "FLAGOS_SDPA_MATH_ATTN", "FLAGOS_A1_IMPL",
            ],
        },
        "baseline": baseline,
        "inject": inj,
        "hit_files": hits,
        "hit_files_stale_excluded": stale,
        "hit_attribution": attr,
        "hit_attribution_check": {
            "hits_total": hits_total,
            "phase_attributed_total": phase_total,
            "consistent": hits_total == phase_total,
            "note": "phase_attributed 含 outside_windows 桶（如实计入）",
        },
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "forward_hits": forward_hits,
        "total_hits_all_incl_probe": hits_total,
        "outputs_equal": (
            [o["token_ids"] for o in baseline.get("outputs") or []]
            == [o["token_ids"] for o in inj.get("outputs") or []]
            if baseline.get("ok") and inj.get("ok") else None),
        "correctness": correctness,
        "perf": perf,
        "acceptance": acceptance,
        "c2_status": None,
        "notes": [],
    }
    ok_run = all(results[m].get("ok") for m in MODES
                 if not results[m].get("skipped"))
    if ok_run:
        if (forward_hits or 0) > 0:
            summary["c2_status"] = "c2-hit"
            summary["notes"].append(
                "真实 vLLM-Ascend 前向命中 aten::embedding。rc/ok 仅表示"
                "运行成功，不等于正确性通过或 C2 验收通过。")
        else:
            summary["c2_status"] = "c2-routing-boundary"
            summary["notes"].append(
                "注入跑完成但 forward_hits=0：aten::embedding 未在真实前向"
                "被调用（或注册未传播到 EngineCore/Worker）。")
        if acceptance["status"] == "PARTIAL":
            summary["notes"].append(
                "验收状态 PARTIAL: 真实引擎已尝试、命中待复核/正确性或性能"
                "未闭环——按 #38 意见 6 不写「C2 验收通过」。")
        if correctness.get("correctness_verdict") in ("INCONCLUSIVE", "FAIL"):
            summary["notes"].append(
                f"同路径正确性 {correctness['correctness_verdict']}: 见 "
                "correctness.per_prompt（首分歧位置）继续排查（意见 2）。")
    else:
        summary["c2_status"] = "c2-failed"
        summary["notes"].append("至少一组跑失败，见 *.stderr.log")

    # 意见 4: 验收判定与退出码收敛为纯函数（scripts/c2_acceptance.decide）。
    decision = decide(summary, mode=args.mode)
    summary["acceptance_decision"] = decision
    summary["run_mode"] = args.mode

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "run_id": run_id,
        "run_mode": args.mode,
        "c2_status": summary["c2_status"],
        "acceptance": acceptance,
        "acceptance_decision": {
            "mode": decision["mode"],
            "collection_status": decision["collection_status"],
            "acceptance_status": decision["acceptance_status"],
            "failed_checks": decision["failed_checks"],
            "warnings": decision["warnings"],
            "exit_code": decision["exit_code"],
        },
        "forward_hits": forward_hits,
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "phase_attribution": attr["by_role"],
        "stale_excluded": len(stale),
        "correctness": {
            "verdict": correctness.get("correctness_verdict"),
            "identity": correctness.get("identity_pass_count"),
            "first2": correctness.get("first2_match"),
        },
        "perf_gate": perf["gate_verdict"],
        "perf_ratios": {c["name"]: c.get("ratio_median")
                        for c in perf["comparisons"]},
        "a1_reg_info": inject.get("a1_reg_info"),
        "out": str(out_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))
    return decision["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
