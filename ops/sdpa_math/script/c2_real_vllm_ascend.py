#!/usr/bin/env python3
"""C2 真实 vLLM-Ascend 双跑 harness（issue #38）。

基线: 无 FlagOS 注入的 vLLM-Ascend 前向。
注入: sitecustomize（含 EngineCore 子进程）+ FLAGOS_A1_OPS=sdpa_math，
      按 pid 分片写调用计数（docs/testing.md）。
断言: 双跑均完成；记录 token ids / 版本 / 模型 / commit / 命中计数。
      total_hits=0 且 probe>0 时记为 routing boundary（known-issues #19），
      不把默认路由声明为 C2 命中。

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
    # 注入模式**不**把 injection/ 放进 EngineCore 可见的 sitecustomize
    # 路径之外的自动加载——NPU 上 sitecustomize 在 fork 子进程里
    # 触发 torch.npu 再初始化会失败（见报告 routing/init 边界）。
    # 主进程探针在 child runner 内显式注册，不依赖 sitecustomize。
    parts = ["/vllm-workspace/vllm-ascend", *_CANN]
    if not inject:
        # 基线: 仍暴露 injection 目录但不设 FLAGOS_A1_ATEN_INJECT，
        # 保证 PYTHONPATH 形态与注入跑对称。
        parts = [str(ROOT / "injection")] + parts
    env["PYTHONPATH"] = ":".join(parts + [p for p in [env.get("PYTHONPATH")] if p])
    env["VLLM_PLUGINS"] = "ascend"
    env["ASCEND_HOME_PATH"] = "/usr/local/Ascend/cann-9.0.0"
    env["HCCL_VERIFY"] = "0"
    if inject:
        env["FLAGOS_A1_ATEN_INJECT"] = "0"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["FLAGOS_A1_OPS"] = "sdpa_math"
        env["A1_DISPATCH_KEY"] = "AutogradPrivateUse1"
        env["FLAGOS_C2_COUNT_DIR"] = str(workdir)
    else:
        for k in ("FLAGOS_A1_ATEN_INJECT", "FLAGOS_A1_OPS",
                  "FLAGOS_C2_COUNT_DIR", "A1_DISPATCH_KEY"):
            env.pop(k, None)
    return env


# 子进程: mode ∈ {baseline, inject}
_CHILD_RUNNER = r"""
import json, os, pathlib, sys, time, traceback

mode, model, prompt, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
result = {"mode": mode, "ok": False}
dump = None

try:
    import torch
    import torch_npu  # noqa: F401
    result["torch"] = torch.__version__
    result["torch_npu"] = getattr(__import__("torch_npu"), "__version__", "?")
    result["npu_available"] = bool(hasattr(torch, "npu") and torch.npu.is_available())
    result["npu_count"] = torch.npu.device_count() if result["npu_available"] else 0

    if mode == "inject":
        # 主进程: 隔离加载 + 带 counter 注册。
        # 注意: 此处**不**立即在 NPU 上跑张量——父进程过早初始化 NPU
        # 会使随后 fork 的 EngineCore 报 "Cannot re-initialize NPU"。
        # probe 放到 generate 之后。
        root = pathlib.Path(os.environ["FLAGOS_TEMPLATES_ROOT"])
        sys.path.insert(0, str(root / "injection"))
        from a1_aten_ops import _load_isolated, _maybe_counter
        r = _load_isolated(root, "sdpa_math")
        if r is None:
            raise RuntimeError("sdpa_math isolated load failed")
        mod, _saved = r
        counter, dump = _maybe_counter("sdpa_math")
        libs = mod.register_a1("AutogradPrivateUse1", counter=counter)
        result["registered_libs"] = len(libs)

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

    if mode == "inject":
        # 真实前向结束后的计数（不含下方 probe）
        result["forward_hits_before_probe"] = counter["n"]
        if dump is not None:
            dump()
        # generate 之后再 probe: EngineCore 已完成 fork，父进程此时
        # 在 NPU 上直调 math 后端，验证 A1 拦截链在本进程生效。
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
             timeout_s: int = 300) -> dict:
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
    # EngineCore 侧（若 sitecustomize 生效）+ 主进程 probe 的合计。
    # 判定 C2 命中只看 forward_hits_before_probe / EngineCore 文件，
    # 不把主进程 probe 算进真实前向。
    engine_core_hits = sum(
        int(h.get("hits") or 0) for h in hits
        if h.get("pid") != inject.get("pid"))
    # 主进程计数文件含 probe；用 inject.forward_hits_before_probe 更准
    forward_hits = inject.get("forward_hits_before_probe")
    if forward_hits is None:
        forward_hits = engine_core_hits
    total_hits_all = sum(int(h.get("hits") or 0) for h in hits)
    probe_hits = inject.get("probe_hits_after_direct_call")

    summary = {
        "schema": 1,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": "ascend910",
        "engine": "vllm-ascend",
        "commit": commit,
        "model": model,
        "prompt": PROMPT,
        "python": platform.python_version(),
        "baseline": baseline,
        "inject": inject,
        "hit_files": hits,
        "forward_hits": forward_hits,
        "total_hits_all_incl_probe": total_hits_all,
        "probe_hits_after_direct_call": probe_hits,
        "outputs_equal": (
            baseline.get("token_ids") == inject.get("token_ids")
            if baseline.get("ok") and inject.get("ok") else None),
        "c2_status": None,
        "notes": [],
    }
    if baseline.get("ok") and inject.get("ok"):
        if (forward_hits or 0) > 0:
            summary["c2_status"] = "c2-hit"
            summary["notes"].append(
                "真实 vLLM-Ascend 前向命中 math 后端，C2 成立")
        elif (probe_hits or 0) > 0:
            summary["c2_status"] = "c2-routing-boundary"
            summary["notes"].append(
                "A1 注册链在 NPU 进程内可用（probe 命中），但真实前向 "
                "forward_hits=0：注意力经 npu_fusion_attention，不调用 "
                "aten math 后端（known-issues #19）。不把默认路由声明为 "
                "C2 命中。另: sitecustomize 全进程注入在 NPU fork "
                "EngineCore 上触发 torch.npu 再初始化失败，本 harness "
                "仅做主进程 probe（见 reports/c2_ascend910.md）。")
        else:
            summary["c2_status"] = "c2-inject-unverified"
            summary["notes"].append("注入进程 probe 未命中，注册链存疑")
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
        "probe_hits": probe_hits,
        "outputs_equal": summary["outputs_equal"],
        "baseline_ok": baseline.get("ok"),
        "inject_ok": inject.get("ok"),
        "out": str(out_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["c2_status"] != "c2-failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
