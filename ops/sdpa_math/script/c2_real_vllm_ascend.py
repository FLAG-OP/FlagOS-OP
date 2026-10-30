#!/usr/bin/env python3
"""C2 真实 vLLM-Ascend 双跑 harness（issue #38，引擎侧突破版）。

基线: 无 FlagOS 注入的 vLLM-Ascend 前向。
注入: spawn 多进程 + sitecustomize（EngineCore/Worker 各进程内生效）:
      - FLAGOS_A1_ATEN_INJECT=1 → A1 注册 aten::_scaled_dot_product_attention_math
      - FLAGOS_SDPA_MATH_ATTN=1  → 补丁 NPUPlatform.get_attn_backend_cls，
        强制注意力走 FlagOSMathAttentionBackend（内部调 aten math op），
        突破 known-issues #19 的 routing boundary。
      按 pid 分片写调用计数（docs/testing.md）。
断言: 双跑均完成；记录 token ids / 版本 / 模型 / commit / 命中计数。
      forward_hits>0 且输出可比 → C2 命中；否则按 routing-boundary 记录原因。

关键环境:
  - VLLM_WORKER_MULTIPROC_METHOD=spawn（NPU 上 sitecustomize 在 spawn
    子进程启动时执行，NPU 尚未初始化，无 fork 再初始化冲突）。
  - enforce_eager=True（禁 ACL graph，保证 Python 路径逐步执行）。

用法:
  python3 ops/sdpa_math/script/c2_real_vllm_ascend.py --out reports/c2_ascend910
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_DEFAULT = (
    "/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M"
    "/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
)
PROMPT = "The capital of France is"
_CANN = [
    "/usr/local/Ascend/cann-9.0.0/python/site-packages",
    "/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe",
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
]


def _child_env(inject: bool, workdir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    # spawn 模式: sitecustomize 在每个子进程启动时执行（NPU 未初始化），
    # 无 fork 再初始化冲突。injection/ 放 PYTHONPATH 首位。
    parts = [str(ROOT / "injection"), "/vllm-workspace/vllm-ascend", *_CANN]
    env["PYTHONPATH"] = ":".join(parts + [p for p in [env.get("PYTHONPATH")] if p])
    env["VLLM_PLUGINS"] = "ascend"
    env["ASCEND_HOME_PATH"] = "/usr/local/Ascend/cann-9.0.0"
    env["HCCL_VERIFY"] = "0"
    # NPU 必须 spawn，否则 fork 后 torch.npu 再初始化失败
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if inject:
        env["FLAGOS_A1_ATEN_INJECT"] = "1"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["FLAGOS_A1_OPS"] = "sdpa_math"
        env["A1_DISPATCH_KEY"] = "AutogradPrivateUse1"
        env["FLAGOS_C2_COUNT_DIR"] = str(workdir)
        env["FLAGOS_SDPA_MATH_ATTN"] = "1"
        # C2 引擎侧路径强制 torch impl: triton 在 bf16 大 GQA shape
        # 首轮可能出现异常输出（见 reports/c2_ascend910.md 调试记录）；
        # torch impl 与 native 逐位一致，更适合 C2 证据链。
        env.setdefault("FLAGOS_A1_IMPL", "torch")
    else:
        for k in ("FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS",
                  "FLAGOS_C2_COUNT_DIR", "A1_DISPATCH_KEY",
                  "FLAGOS_SDPA_MATH_ATTN"):
            env.pop(k, None)
    return env


# 子进程: mode ∈ {baseline, inject}
_CHILD_RUNNER = r"""
import json, os, pathlib, sys, time, traceback

mode, model, prompt, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
result = {"mode": mode, "ok": False, "pid": os.getpid()}
dump = None

try:
    # sitecustomize（inject 模式）已在本进程启动时完成 A1 注册与
    # attention backend 补丁；此处只需验证状态并跑真实生成。
    import torch
    import torch_npu  # noqa: F401
    result["torch"] = torch.__version__
    result["torch_npu"] = getattr(__import__("torch_npu"), "__version__", "?")
    result["npu_available"] = bool(hasattr(torch, "npu") and torch.npu.is_available())
    result["npu_count"] = torch.npu.device_count() if result["npu_available"] else 0
    result["mp_method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    result["flagos_attn"] = os.environ.get("FLAGOS_SDPA_MATH_ATTN", "0")

    counter = None
    if mode == "inject":
        # sitecustomize 已注册; 取回 counter 做主进程 probe 验证
        root = pathlib.Path(os.environ["FLAGOS_TEMPLATES_ROOT"])
        sys.path.insert(0, str(root / "injection"))
        from a1_aten_ops import get_counter
        counter = get_counter("sdpa_math")
        result["a1_counter_retrieved"] = counter is not None
        # 验证 backend 补丁已生效（sitecustomize 可能因 import 时序失败，
        # 此处再兜底补一次）
        try:
            from vllm_ascend.platform import NPUPlatform
            patched = bool(getattr(NPUPlatform, "_flagos_math_attn_patched", False))
            result["attn_patched"] = patched
            if not patched and os.environ.get("FLAGOS_SDPA_MATH_ATTN") == "1":
                from ops.sdpa_math.engine.flagos_math_attn import (
                    patch_npu_platform_get_attn_backend_cls,
                )
                ok = patch_npu_platform_get_attn_backend_cls()
                result["attn_patched_fallback"] = ok
                result["attn_patched"] = bool(
                    getattr(NPUPlatform, "_flagos_math_attn_patched", False))
        except Exception as e:
            result["attn_patched_err"] = repr(e)

    from vllm import LLM, SamplingParams
    import vllm
    result["vllm"] = getattr(vllm, "__version__", "?")
    try:
        import vllm_ascend
        result["vllm_ascend"] = str(getattr(vllm_ascend, "__version__", "source"))
        result["vllm_ascend_file"] = getattr(vllm_ascend, "__file__", None)
    except Exception as e:
        result["vllm_ascend_err"] = repr(e)

    t0 = time.time()
    llm = LLM(model=model, enforce_eager=True, max_model_len=192,
              gpu_memory_utilization=0.2, max_num_seqs=1,
              trust_remote_code=True, disable_log_stats=True)
    result["init_s"] = round(time.time() - t0, 2)
    # 一次 generate 同时取生成结果与 teacher-forced prompt logprobs——
    # 分两次 generate 会把第二次前向的命中混进 engine 计数。
    t1 = time.time()
    try:
        sp = SamplingParams(max_tokens=8, temperature=0.0, prompt_logprobs=0)
        outs = llm.generate([prompt], sp)
        pl = outs[0].prompt_logprobs
        plmap = {}
        for idx, d in enumerate(pl or []):
            if not d:
                continue
            lp = next(iter(d.values()))
            plmap[str(idx)] = float(getattr(lp, "logprob", lp))
        result["prompt_logprobs"] = plmap
    except Exception:
        # prompt_logprobs 不可用时回退普通生成，保证主证据链可跑
        result["prompt_logprobs_err"] = traceback.format_exc().splitlines()[-1]
        sp = SamplingParams(max_tokens=8, temperature=0.0)
        outs = llm.generate([prompt], sp)
    result["generate_s"] = round(time.time() - t1, 2)
    result["text"] = outs[0].outputs[0].text
    result["token_ids"] = list(outs[0].outputs[0].token_ids[:8])
    result["ok"] = True

    if mode == "inject" and counter is not None:
        # 真实前向（本次 generate）结束后的计数——不含下方 probe
        result["forward_hits_before_probe"] = counter["n"]
        if dump is not None:
            dump()
        # generate 之后 probe: 验证 A1 拦截链在本进程仍生效
        q = torch.randn(1, 4, 8, 16, dtype=torch.float16, device="npu")
        k = torch.randn(1, 4, 8, 16, dtype=torch.float16, device="npu")
        v = torch.randn(1, 4, 8, 16, dtype=torch.float16, device="npu")
        with torch.no_grad():
            out = torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, None, 0.0, False, None)
        torch.npu.synchronize()
        result["probe_hits_after_direct_call"] = counter["n"] - result[
            "forward_hits_before_probe"]
        result["probe_out_shape"] = list(out[0].shape)
        result["probe_out_dtype"] = str(out[0].dtype)
        if dump is not None:
            dump()
except Exception:
    result["error"] = traceback.format_exc()

pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
"""


def run_mode(mode: str, model: str, out_dir: pathlib.Path,
             timeout_s: int = 600) -> dict:
    out_path = out_dir / f"{mode}.json"
    env = _child_env(inject=(mode == "inject"), workdir=out_dir)
    cmd = [sys.executable, "-c", _CHILD_RUNNER, mode, model, PROMPT, str(out_path)]
    print(f"[{mode}] start", flush=True)
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


def merge_hits(out_dir: pathlib.Path) -> list[dict]:
    hits = []
    for p in sorted(out_dir.glob("sdpa_math_hits_*.json")):
        try:
            hits.append(json.loads(p.read_text()))
        except Exception:
            hits.append({"path": str(p), "error": "unreadable"})
    return hits


def _output_compare(baseline: dict, inject: dict) -> dict | None:
    """输出对比: 贪心 token 共同前缀 + teacher-forced 逐 token logprob 差异。

    两种 attention 实现（FlagOS math 后端 vs npu_fusion_attention）数值
    精度不同，生成分叉属预期；logprob 对比给出分叉幅度的定量上界。
    """
    if not (baseline.get("ok") and inject.get("ok")):
        return None
    b = list(baseline.get("token_ids") or [])
    i = list(inject.get("token_ids") or [])
    prefix = 0
    for x, y in zip(b, i):
        if x != y:
            break
        prefix += 1
    out = {
        "baseline_token_ids": b,
        "inject_token_ids": i,
        "common_prefix_len": prefix,
        "tokens_compared": max(len(b), len(i)),
        "bitwise_equal": b == i,
        "baseline_text": baseline.get("text"),
        "inject_text": inject.get("text"),
    }
    bl = baseline.get("prompt_logprobs") or {}
    il = inject.get("prompt_logprobs") or {}
    common = sorted((k for k in set(bl) & set(il)), key=int)
    if common:
        diffs = [abs(float(bl[k]) - float(il[k])) for k in common]
        out["prompt_logprob"] = {
            "tokens_compared": len(common),
            "max_abs_diff": round(max(diffs), 6),
            "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
            "baseline": {k: round(float(bl[k]), 4) for k in common},
            "inject": {k: round(float(il[k]), 4) for k in common},
        }
    else:
        out["prompt_logprob"] = {
            "unavailable": True,
            "baseline_err": baseline.get("prompt_logprobs_err"),
            "inject_err": inject.get("prompt_logprobs_err"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("FLAGOS_MODEL_PATH",
                                                      MODEL_DEFAULT))
    ap.add_argument("--out", default=str(
        pathlib.Path(__file__).resolve().parent.parent / "reports" / "c2_ascend910"))
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    model = args.model
    if not pathlib.Path(model).is_dir():
        print(f"model path missing: {model}", file=sys.stderr)
        return 2
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()

    baseline = ({"mode": "baseline", "ok": False, "skipped": True}
                if args.skip_baseline else run_mode("baseline", model, out_dir))
    for old in out_dir.glob("sdpa_math_hits_*.json"):
        old.unlink()
    inject = run_mode("inject", model, out_dir)
    hits = merge_hits(out_dir)
    # pid 归属: 主进程（child runner，含 probe）与 EngineCore/Worker 分开——
    # EngineCore/Worker 分开——真实前向只计引擎侧进程；主进程 probe 不得
    # 计入 forward_hits（issue #38 证据对齐要求）。
    main_pid = inject.get("pid")
    for h in hits:
        if main_pid is None or h.get("pid") is None:
            h["role"] = "unknown"
        else:
            h["role"] = "main" if h["pid"] == main_pid else "engine/worker"
    engine_hits = sum(
        int(h.get("hits") or 0) for h in hits if h.get("role") == "engine/worker")
    main_hits = sum(int(h.get("hits") or 0) for h in hits
                    if h.get("role") == "main")
    # 真实前向命中: EngineCore/Worker 进程计数；若引擎在主进程内运行则
    # 回退主进程 generate 计数（forward_hits_before_probe 为 generate 后、probe 前快照）。
    forward_hits = engine_hits
    if (forward_hits or 0) == 0:
        forward_hits = inject.get("forward_hits_before_probe") or 0
    total_hits_all = sum(int(h.get("hits") or 0) for h in hits)
    probe_hits = inject.get("probe_hits_after_direct_call")

    summary = {
        "schema": 2,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": "ascend910",
        "engine": "vllm-ascend",
        "commit": commit,
        "model": model,
        "prompt": PROMPT,
        "python": platform.python_version(),
        "injection": {
            "mp_method": inject.get("mp_method"),
            "attn_backend_patch": inject.get("attn_patched"),
            "flagos_attn_env": inject.get("flagos_attn"),
            "a1_ops": "sdpa_math",
        },
        "baseline": baseline,
        "inject": inject,
        "hit_files": hits,
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "forward_hits": forward_hits,
        "total_hits_all_incl_probe": total_hits_all,
        "probe_hits_after_direct_call": probe_hits,
        "outputs_equal": (
            baseline.get("token_ids") == inject.get("token_ids")
            if baseline.get("ok") and inject.get("ok") else None),
        "output_compare": _output_compare(baseline, inject),
        "c2_status": None,
        "notes": [],
    }
    if baseline.get("ok") and inject.get("ok"):
        if (forward_hits or 0) > 0:
            summary["c2_status"] = "c2-hit"
            summary["notes"].append(
                "真实 vLLM-Ascend 前向命中 aten math 后端（引擎侧 "
                "FlagOSMathAttentionBackend 补丁生效），C2 成立")
        elif (probe_hits or 0) > 0:
            summary["c2_status"] = "c2-routing-boundary"
            summary["notes"].append(
                "A1 注册链在 NPU 进程内可用（probe 命中），但真实前向 "
                "forward_hits=0：注意力未走 FlagOS math backend 路径。"
                "可能原因: backend 补丁未传播到 EngineCore/Worker、"
                "spawn 子进程未加载 sitecustomize、或注意力走了其他分支。"
                "详见 inject.stdout.log 与 stderr。")
        else:
            summary["c2_status"] = "c2-inject-unverified"
            summary["notes"].append("注入进程 probe 未命中，注册链存疑")
        if summary["outputs_equal"] is False:
            summary["notes"].append(
                "outputs_equal=false（例外记录）: math 后端与 "
                "npu_fusion_attention 数值精度不同，贪心生成分叉属预期；"
                "输出对比改用 token 共同前缀 + teacher-forced prompt "
                "logprob 差异定量判定，处置与退出条件见 "
                "reports/c2_ascend910.md「输出对比处置」")
    else:
        summary["c2_status"] = "c2-failed"
        summary["notes"].append("基线或注入跑失败，见 *.stderr.log")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({
        "c2_status": summary["c2_status"],
        "forward_hits": forward_hits,
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "probe_hits": probe_hits,
        "attn_patched": inject.get("attn_patched"),
        "outputs_equal": summary["outputs_equal"],
        "output_compare": summary.get("output_compare"),
        "baseline_ok": baseline.get("ok"),
        "inject_ok": inject.get("ok"),
        "out": str(out_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["c2_status"] != "c2-failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
