#!/usr/bin/env python3
"""One EngineCore (multiprocess) run for C2 evidence.

Injection + phase-count hook live in ./sitecustomize.py, which is put on
PYTHONPATH *before* importing vllm so the vLLM-spawned EngineCore child loads it.

Usage:
  python3 scripts/c2_enginecore/run.py --op empty_like --mode full \
      --model /tmp/flagos_tiny_llama --count-dir /tmp/c2/empty_like/full \
      --run-id <id>
  # baseline:
  python3 scripts/c2_enginecore/run.py --op __baseline__ --mode full ...
  # init-only (no generate):
  python3 scripts/c2_enginecore/run.py --op empty_like --mode init ...

Prints one line: C2_RUN {json}
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--mode", choices=["init", "full"], default="full")
    ap.add_argument("--model", required=True)
    ap.add_argument("--count-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--platform", default="ascend")
    ap.add_argument("--dispatch-key", default="AutogradPrivateUse1")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--perf-warmup", type=int, default=0,
                    help="warmup generations before timing (full mode)")
    ap.add_argument("--perf-iters", type=int, default=0,
                    help="measured generations; zero disables timing")
    args = ap.parse_args()

    # --- env must be set BEFORE importing vllm (spawned child inherits it) ---
    os.environ["FLAGOS_C2_RUN_ID"] = args.run_id
    os.environ["FLAGOS_A1_OPS"] = "" if args.op == "__baseline__" else args.op
    os.environ["A1_PLATFORM"] = args.platform
    os.environ["A1_DISPATCH_KEY"] = args.dispatch_key
    os.environ["FLAGOS_C2_COUNT_DIR"] = args.count_dir
    phase_file = str(Path(args.count_dir).parent / "phase.txt")
    os.environ["FLAGOS_C2_PHASE_FILE"] = phase_file
    os.environ["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("TASK_QUEUE_ENABLE", "0")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    parts = [str(HERE), str(ROOT)]
    stub = "/workspace/flagrt_stub"
    if Path(stub).is_dir():
        parts.append(stub)
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tokenizer=args.model, enforce_eager=True,
              max_model_len=256, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, seed=0,
              disable_log_stats=True)
    Path(phase_file).write_text("init", encoding="utf-8")
    ids = []
    performance = {"status": "not-collected", "warmup": 0, "iters": 0,
                   "samples_ms": []}
    if args.mode == "full":
        prompts = ["the future of AI is"]
        sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        Path(phase_file).write_text("generation", encoding="utf-8")
        out = llm.generate(prompts, sampling)
        ids = list(out[0].outputs[0].token_ids)
        if args.perf_iters < 0 or args.perf_warmup < 0:
            raise ValueError("performance iteration counts must be non-negative")
        if args.perf_iters:
            for _ in range(args.perf_warmup):
                llm.generate(prompts, sampling)
            samples = []
            for _ in range(args.perf_iters):
                sync = getattr(getattr(torch, "npu", None), "synchronize", None)
                if sync:
                    sync()
                start = time.perf_counter()
                llm.generate(prompts, sampling)
                if sync:
                    sync()
                samples.append((time.perf_counter() - start) * 1000.0)
            performance = {
                "status": "pass", "warmup": args.perf_warmup,
                "iters": args.perf_iters,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.fmean(samples),
                "min_ms": min(samples), "max_ms": max(samples),
            }
        Path(phase_file).write_text("done", encoding="utf-8")
    print("C2_RUN " + json.dumps({
        "op": args.op, "mode": args.mode, "run_id": args.run_id,
        "ids": ids, "model": args.model, "performance": performance,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
