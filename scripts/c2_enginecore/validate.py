#!/usr/bin/env python3
"""C2 acceptance for one op via real EngineCore (multiprocess) double-run.

Runs three EngineCore processes:
  baseline   : no registration, full generate
  init-only  : registration, LLM init only (no generate) -> negative control
  injected   : registration, full generate

Collects per-PID dumps from sitecustomize.py (init/generation/final counts and
execute_model phase deltas), then `decide()` enforces:
  - all runs rc == 0
  - non-empty baseline/injected token ids, and they match
  - EngineCore phase dump present (execute_model observed)
  - generation-phase hits > 0  (init-only hits do NOT count)

Exit code is non-zero on any failure. Usage:
  python3 scripts/c2_enginecore/validate.py --op empty_like \
      --model /tmp/flagos_tiny_llama --outdir reports/c2_ascend910
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata as md
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "run.py"


def decide(s: dict) -> tuple[bool, list[str]]:
    """Pure acceptance decision (unit-tested with negative cases)."""
    reasons = []
    if s.get("baseline_rc") != 0:
        reasons.append("baseline run failed (rc != 0)")
    if s.get("injected_full_rc") != 0:
        reasons.append("injected full run failed (rc != 0)")
    if s.get("injected_init_rc") != 0:
        reasons.append("injected init run failed (rc != 0)")
    if not s.get("baseline_ids"):
        reasons.append("baseline produced empty token ids")
    if not s.get("injected_ids"):
        reasons.append("injected produced empty token ids")
    if s.get("baseline_ids") != s.get("injected_ids"):
        reasons.append("baseline/injected token ids mismatch")
    if not s.get("has_full_dump"):
        reasons.append("no EngineCore phase dump (execute_model not observed)")
    if (s.get("generation_hits") or 0) <= 0:
        reasons.append("generation-phase hits == 0 (init-only or not consumed)")
    return (not reasons), reasons


def _agg(count_dir: Path, op: str) -> dict:
    recs = []
    for f in glob.glob(str(count_dir / "*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:  # noqa: BLE001
            pass
    init = sum(r.get("init_count", {}).get(op, 0) for r in recs)
    gen = sum(r.get("generation_hits", {}).get(op, 0) for r in recs)
    final = sum(r.get("final_count", {}).get(op, 0) for r in recs)
    exec_calls = sum(int(r.get("execute_calls", 0)) for r in recs)
    return {"init": init, "gen": gen, "final": final,
            "exec_calls": exec_calls, "pids": [r.get("pid") for r in recs]}


def _run(op: str, mode: str, model: str, count_dir: Path,
         run_id: str, log: Path, max_tokens: int, gpu_util: float) -> tuple[int, list]:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = env.get("ASCEND_RT_VISIBLE_DEVICES", "0")
    cmd = [sys.executable, str(RUN), "--op", op, "--mode", mode,
           "--model", model, "--count-dir", str(count_dir),
           "--run-id", run_id, "--max-tokens", str(max_tokens),
           "--gpu-util", str(gpu_util)]
    with open(log, "w") as lf:
        p = subprocess.run(cmd, env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    ids = []
    try:
        for line in log.read_text().splitlines():
            if line.startswith("C2_RUN "):
                ids = json.loads(line[len("C2_RUN "):]).get("ids", [])
    except Exception:  # noqa: BLE001
        pass
    return p.returncode, ids


def _env() -> dict:
    def v(p):
        try:
            return md.version(p)
        except Exception:  # noqa: BLE001
            return None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {
        "torch": v("torch"), "torch_npu": v("torch_npu"),
        "flagtree": v("flagtree"), "flag_gems": v("flag-gems"),
        "vllm": v("vllm"), "vllm_plugin_fl": v("vllm-plugin-fl"),
        "python": sys.version.split()[0],
        "ascend_rt_visible_devices": os.environ.get(
            "ASCEND_RT_VISIBLE_DEVICES", "0"),
        "git_commit": commit,
        "note": "vllm-plugin-fl / PlatformFL stack (not vLLM-Ascend)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default=str(ROOT / "reports" / "c2_ascend910"))
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    base = Path(args.outdir) / args.op
    base.mkdir(parents=True, exist_ok=True)
    c_base = base / f"counts_baseline_{run_id}"
    c_init = base / f"counts_init_{run_id}"
    c_full = base / f"counts_full_{run_id}"
    for d in (c_base, c_init, c_full):
        d.mkdir(parents=True, exist_ok=True)

    rc_b, ids_b = _run("__baseline__", "full", args.model, c_base, run_id,
                       base / "baseline.stdout.log", args.max_tokens,
                       args.gpu_util)
    # init-only registers the target op but never calls generate()
    rc_i, ids_i = _run(args.op, "init", args.model, c_init, run_id,
                       base / "init.stdout.log", args.max_tokens, args.gpu_util)
    rc_f, ids_f = _run(args.op, "full", args.model, c_full, run_id,
                       base / "inject.stdout.log", args.max_tokens,
                       args.gpu_util)

    agg_init = _agg(c_init, args.op)
    agg_full = _agg(c_full, args.op)
    summary = {
        "schema": 1,
        "kind": "c2_enginecore",
        "op": args.op,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "env": _env(),
        "baseline_rc": rc_b, "injected_full_rc": rc_f,
        "injected_init_rc": rc_i,
        "baseline_ids": ids_b, "injected_ids": ids_f,
        "init_ids": ids_i,
        "init_only": agg_init,
        "injected": agg_full,
        "init_hits": agg_full["init"],
        "generation_hits": agg_full["gen"],
        "final_hits": agg_full["final"],
        "has_full_dump": agg_full["exec_calls"] > 0,
        "logs": {
            "baseline": str(base / "baseline.stdout.log"),
            "init": str(base / "init.stdout.log"),
            "inject": str(base / "inject.stdout.log"),
        },
    }
    ok, reasons = decide(summary)
    summary["passed"] = ok
    summary["reasons"] = reasons
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"op": args.op, "passed": ok, "reasons": reasons,
                      "baseline_ids": ids_b, "injected_ids": ids_f,
                      "init_hits": summary["init_hits"],
                      "generation_hits": summary["generation_hits"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
