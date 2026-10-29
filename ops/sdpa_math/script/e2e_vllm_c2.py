#!/usr/bin/env python3
"""C2 evidence: real vLLM EngineCore consumes aten::_scaled_dot_product_attention_math.

Default vLLM attention (flag_gems flash_attn_varlen / FA3) does **not** call the
math composite.  This dual-run therefore uses an explicit model path:
model_impl=transformers + sitecustomize replaces ALL_ATTENTION_FUNCTIONS["vllm"]
with F.sdpa under SDPBackend.MATH, which dispatches to
aten::_scaled_dot_product_attention_math (A1 counter intercepts it).

Dual-run in separate subprocesses (A1 registrations cannot be undone):
  1. baseline  — real LLM.generate with MATH path, **no** A1
  2. injected  — same MATH path + FLAGOS_C2_SDPA_MATH_INJECT=1 + counter file

PASS criteria:
  - both phases complete
  - injected phase: total math-op hits > 0 across count shards
  - output token ids match between baseline and injected (same path/weights)
Evidence JSON is written under ops/sdpa_math/reports/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent

CHILD_ENV_EXTRA = {
    "VLLM_LOGGING_LEVEL": "ERROR",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}

_CHILD_SCRIPT = r'''
import glob, json, os, sys
from pathlib import Path


def main() -> int:
    model = os.environ["C2_MODEL"]
    prompt_ids = json.loads(os.environ["C2_PROMPT_IDS"])
    max_tokens = int(os.environ.get("C2_MAX_TOKENS", "8"))
    seed = int(os.environ.get("C2_SEED", "20260823"))
    out_path = os.environ["C2_OUT_JSON"]
    count_base = os.environ.get("A1_SDPA_MATH_COUNT_FILE", "")
    model_impl = os.environ.get("C2_MODEL_IMPL", "transformers")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        model_impl=model_impl,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(os.environ.get("C2_GPU_UTIL", "0.35")),
        max_model_len=int(os.environ.get("C2_MAX_MODEL_LEN", "256")),
        seed=seed,
        load_format="safetensors",
        enable_prefix_caching=False,
    )
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        seed=seed,
        detokenize=False,
    )
    outs = llm.generate([{"prompt_token_ids": prompt_ids}], params)
    token_ids = [list(o.outputs[0].token_ids) for o in outs]
    prompt_token_ids = [list(o.prompt_token_ids) for o in outs]

    hits = {}
    if count_base:
        for p in glob.glob(f"{count_base}.*"):
            try:
                data = json.loads(Path(p).read_text())
            except Exception:
                continue
            for k, v in data.items():
                hits[k] = hits.get(k, 0) + int(v)

    payload = {
        "token_ids": token_ids,
        "prompt_token_ids": prompt_token_ids,
        "aten_hits": hits,
        "pid": os.getpid(),
        "model": model,
        "model_impl": model_impl,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    print("C2_CHILD_OK", json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _run_child(env_extra: dict, out_json: Path, tag: str) -> dict:
    env = os.environ.copy()
    env.update(CHILD_ENV_EXTRA)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "injection"), str(ROOT)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, prefix=f"c2_sdpa_{tag}_"
    ) as f:
        f.write(_CHILD_SCRIPT)
        script = f.name
    try:
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, script],
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("C2_TIMEOUT", "600")),
        )
        elapsed = time.time() - t0
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-8000:])
        raise RuntimeError(f"{tag} child exit={proc.returncode}")
    if not out_json.is_file():
        raise RuntimeError(f"{tag} missing evidence {out_json}")
    data = json.loads(out_json.read_text())
    data["elapsed_s"] = round(elapsed, 3)
    data["stderr_tail"] = proc.stderr[-2000:]
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=os.environ.get("C2_MODEL", "/tmp/flagos_tiny_llama"),
        help="HF/vLLM loadable model path",
    )
    ap.add_argument(
        "--prompt-ids",
        default="[1, 5, 6, 7]",
        help="JSON list of prompt token ids",
    )
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument(
        "--gpu-util", type=float, default=0.35,
        help="gpu_memory_utilization for tiny models",
    )
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument(
        "--model-impl",
        default="transformers",
        choices=["transformers", "vllm"],
        help="vLLM model backend (transformers required for math path patch)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OP_DIR / "reports" / "e2e_vllm_c2_thead-ppu-c2.json",
    )
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"SKIP: model not found: {model}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count_base = Path(f"/tmp/flagos_c2_sdpa_math_{os.getpid()}")
    for p in count_base.parent.glob(count_base.name + ".*"):
        try:
            p.unlink()
        except OSError:
            pass

    common = {
        "C2_MODEL": str(model),
        "C2_PROMPT_IDS": args.prompt_ids,
        "C2_MAX_TOKENS": str(args.max_tokens),
        "C2_SEED": str(args.seed),
        "C2_GPU_UTIL": str(args.gpu_util),
        "C2_MAX_MODEL_LEN": str(args.max_model_len),
        "C2_MODEL_IMPL": args.model_impl,
        "FLAGOS_TEMPLATES_ROOT": str(ROOT),
    }

    base_dir = args.out.parent
    base_out = base_dir / f"_c2_sdpa_baseline_{os.getpid()}.json"
    inj_out = base_dir / f"_c2_sdpa_injected_{os.getpid()}.json"

    print("[1/2] baseline MATH path (no A1) ...")
    baseline_env = dict(common)
    baseline_env["C2_OUT_JSON"] = str(base_out)
    baseline_env["FLAGOS_C2_FORCE_MATH_PATH"] = "1"
    baseline = _run_child(baseline_env, base_out, "baseline")
    print(f"  hits={baseline.get('aten_hits')} "
          f"tokens={baseline.get('token_ids')} "
          f"t={baseline['elapsed_s']}s")

    print("[2/2] injected real vLLM (MATH path + A1) ...")
    inj_env = dict(common)
    inj_env.update({
        "C2_OUT_JSON": str(inj_out),
        "FLAGOS_C2_SDPA_MATH_INJECT": "1",
        "A1_DISPATCH_KEY": "CUDA",
        "SDPA_MATH_IMPL": os.environ.get("SDPA_MATH_IMPL", "torch"),
        "A1_SDPA_MATH_COUNT_FILE": str(count_base),
    })
    injected = _run_child(inj_env, inj_out, "injected")
    print(f"  hits={injected.get('aten_hits')} "
          f"tokens={injected.get('token_ids')} "
          f"t={injected['elapsed_s']}s")

    base_tokens = baseline.get("token_ids")
    inj_tokens = injected.get("token_ids")
    math_key = "aten::_scaled_dot_product_attention_math"
    math_hits = int(injected.get("aten_hits", {}).get(math_key, 0))
    tokens_match = base_tokens == inj_tokens
    status = "PASS" if (math_hits > 0 and tokens_match) else "FAIL"

    evidence = {
        "op": "sdpa_math",
        "profile": "thead-ppu-c2",
        "consumer_type": "vllm-enginecore",
        "tier": "C2",
        "dispatch_key": "CUDA",
        "platform": "thead-ppu",
        "model": str(model),
        "model_impl": args.model_impl,
        "explicit_path": (
            "ALL_ATTENTION_FUNCTIONS['vllm'] -> F.sdpa + SDPBackend.MATH "
            "(default flash/flagos path does not consume math composite)"
        ),
        "baseline_path": "same MATH path without A1 (FLAGOS_C2_FORCE_MATH_PATH=1)",
        "prompt_ids": json.loads(args.prompt_ids),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "baseline": {
            "token_ids": base_tokens,
            "prompt_token_ids": baseline.get("prompt_token_ids"),
            "aten_hits": baseline.get("aten_hits"),
            "elapsed_s": baseline.get("elapsed_s"),
        },
        "injected": {
            "token_ids": inj_tokens,
            "prompt_token_ids": injected.get("prompt_token_ids"),
            "aten_hits": injected.get("aten_hits"),
            "elapsed_s": injected.get("elapsed_s"),
            "count_shards": sorted(
                str(p.name)
                for p in count_base.parent.glob(count_base.name + ".*")
            ),
        },
        "assertions": {
            "math_hits_gt_0": math_hits > 0,
            "output_tokens_match": tokens_match,
        },
        "status": status,
        "date": time.strftime("%Y-%m-%d"),
        "commit_hint": "wt-2026-10-26-fix",
    }
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")

    for p in (base_out, inj_out):
        try:
            p.unlink()
        except OSError:
            pass

    print(json.dumps(evidence["assertions"], indent=2))
    print(f"status={status} out={args.out}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
