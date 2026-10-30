#!/usr/bin/env python3
"""C2 真实 vLLM-Ascend 双跑 harness——embedding（issue #38）。

与 sdpa_math/script/c2_real_vllm_ascend.py 同构，但注入目标为
aten::embedding（A1 拦截）。embedding 在 vLLM 每个 token 的 input
embeddings 路径上，真实前向命中率远高于注意力类算子。

基线: 无 FlagOS 注入的 vLLM-Ascend 前向。
注入: spawn 多进程 + sitecustomize（EngineCore/Worker 各进程内生效）:
      FLAGOS_A1_ATEN_INJECT=1 + FLAGOS_A1_OPS=embedding。
      按 pid 分片写调用计数（docs/testing.md）。
断言: 双跑均完成；记录 token ids / 版本 / 模型 / commit / 命中计数。

用法:
  python3 ops/embedding/script/c2_real_vllm_ascend.py --out reports/c2_ascend910
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
    parts = [str(ROOT / "injection"), "/vllm-workspace/vllm-ascend", *_CANN]
    env["PYTHONPATH"] = ":".join(parts + [p for p in [env.get("PYTHONPATH")] if p])
    env["VLLM_PLUGINS"] = "ascend"
    env["ASCEND_HOME_PATH"] = "/usr/local/Ascend/cann-9.0.0"
    env["HCCL_VERIFY"] = "0"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if inject:
        env["FLAGOS_A1_ATEN_INJECT"] = "1"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["FLAGOS_A1_OPS"] = "embedding"
        # 推理（no_grad）路径 dispatch 到 PrivateUse1；
        # AutogradPrivateUse1 已被 vllm_ascend 注册，会冲突。
        env["A1_DISPATCH_KEY"] = "PrivateUse1"
        env["FLAGOS_C2_COUNT_DIR"] = str(workdir)
    else:
        for k in ("FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS",
                  "FLAGOS_C2_COUNT_DIR", "A1_DISPATCH_KEY"):
            env.pop(k, None)
    return env


_CHILD_RUNNER = r"""
import json, os, pathlib, sys, time, traceback

mode, model, prompt, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
result = {"mode": mode, "ok": False, "pid": os.getpid()}

try:
    import torch
    import torch_npu  # noqa: F401
    result["torch"] = torch.__version__
    result["torch_npu"] = getattr(__import__("torch_npu"), "__version__", "?")
    result["npu_available"] = bool(hasattr(torch, "npu") and torch.npu.is_available())
    result["npu_count"] = torch.npu.device_count() if result["npu_available"] else 0
    result["mp_method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "fork")

    counter = None
    if mode == "inject":
        root = pathlib.Path(os.environ["FLAGOS_TEMPLATES_ROOT"])
        sys.path.insert(0, str(root / "injection"))
        from a1_aten_ops import get_counter
        counter = get_counter("embedding")
        result["a1_counter_retrieved"] = counter is not None

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
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    t1 = time.time()
    outs = llm.generate([prompt], sp)
    result["generate_s"] = round(time.time() - t1, 2)
    result["text"] = outs[0].outputs[0].text
    result["token_ids"] = list(outs[0].outputs[0].token_ids[:8])
    result["ok"] = True

    if mode == "inject" and counter is not None:
        result["forward_hits_before_probe"] = counter["n"]
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
    for p in sorted(out_dir.glob("embedding_hits_*.json")):
        try:
            hits.append(json.loads(p.read_text()))
        except Exception:
            hits.append({"path": str(p), "error": "unreadable"})
    return hits


def _output_compare(baseline: dict, inject: dict) -> dict | None:
    """生成结果对比: 共同前缀长度与逐位一致性（issue #38 输出对比项）。"""
    if not (baseline.get("ok") and inject.get("ok")):
        return None
    b = list(baseline.get("token_ids") or [])
    i = list(inject.get("token_ids") or [])
    prefix = 0
    for x, y in zip(b, i):
        if x != y:
            break
        prefix += 1
    return {
        "baseline_token_ids": b,
        "inject_token_ids": i,
        "common_prefix_len": prefix,
        "tokens_compared": max(len(b), len(i)),
        "bitwise_equal": b == i,
        "baseline_text": baseline.get("text"),
        "inject_text": inject.get("text"),
    }


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
    for old in out_dir.glob("embedding_hits_*.json"):
        old.unlink()
    inject = run_mode("inject", model, out_dir)
    hits = merge_hits(out_dir)
    # pid 归属: 主进程（child runner，含 probe）与 EngineCore/Worker 分开——
    # 直读单个 hits 文件时必须能区分角色，否则易把主进程 0 命中误读为"无证据"
    # 或把 probe 计入真实前向（issue #38 证据对齐要求）。
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
    # 真实前向命中: 优先取 EngineCore/Worker 进程计数；若引擎在主进程内
    # 运行（无分片子进程）则回退主进程 generate 计数（不含 probe）。
    forward_hits = engine_hits
    if (forward_hits or 0) == 0:
        forward_hits = inject.get("forward_hits_before_probe") or 0
    total_hits_all = sum(int(h.get("hits") or 0) for h in hits)

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
            "a1_ops": "embedding",
        },
        "baseline": baseline,
        "inject": inject,
        "hit_files": hits,
        "engine_core_hits": engine_hits,
        "main_process_hits": main_hits,
        "forward_hits": forward_hits,
        "total_hits_all_incl_probe": total_hits_all,
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
                "真实 vLLM-Ascend 前向命中 aten::embedding，C2 成立")
        else:
            summary["c2_status"] = "c2-routing-boundary"
            summary["notes"].append(
                "注入跑完成但 forward_hits=0：aten::embedding 未在真实前向"
                "被调用（或 sitecustomize 注册未传播到 EngineCore/Worker）。"
                "详见 inject.stdout.log 与 stderr。")
        if summary["outputs_equal"] is False:
            summary["notes"].append("WARNING: 基线与注入 token_ids 不一致")
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
        "outputs_equal": summary["outputs_equal"],
        "baseline_ok": baseline.get("ok"),
        "inject_ok": inject.get("ok"),
        "out": str(out_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["c2_status"] != "c2-failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
