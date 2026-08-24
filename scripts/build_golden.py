#!/usr/bin/env python3
"""构建黄金输出（golden outputs）。

跨硬件黄金锚点生成器。两种引擎:
  vllm         reference 路径独立跑 N 次（GPU/XPU 设备）
  transformers HuggingFace 直接 CPU 推理（最权威语义参考，绕过
               vLLM/厂商栈; 确定性，默认 1 次快照）
保存全部快照并计算"多数票共识前缀"，供测试/CI 回归比对。

用法:
  python3 scripts/build_golden.py --device p800-kunlunxin --snapshots 3
  python3 scripts/build_golden.py --device cpu --prompts-limit 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN_DIR = ROOT / "golden"


def consensus_prefix(snapshots: list[list[int]], length: int) -> list[int]:
    """逐位置多数票，取前 length 个共识 token。"""
    L = min(min(len(s) for s in snapshots), length)
    out = []
    for k in range(L):
        c = Counter(s[k] for s in snapshots if k < len(s))
        token, _ = c.most_common(1)[0]
        # 该位置若非全员多数，截断（保守）
        if c[token] < len(snapshots):
            break
        out.append(token)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="p800-kunlunxin")
    ap.add_argument("--snapshots", type=int, default=0,
                    help="独立运行次数（0=按引擎自动: vllm 3 次, transformers 1 次）")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "vllm", "transformers"])
    ap.add_argument("--prompts-limit", type=int, default=0,
                    help="限制 prompt 条数（0=全部; CPU 慢时可用）")
    ap.add_argument("--prefix", type=int, default=16,
                    help="共识前缀最大长度")
    args = ap.parse_args()

    from common.device import load_profile
    from scripts.drift_study import run_once

    profile = load_profile(args.device)
    engine = (profile.golden_engine_hint if args.engine == "auto"
              else args.engine)
    n_snap = args.snapshots or (1 if engine == "transformers" else 3)
    GOLDEN_DIR.mkdir(exist_ok=True)

    if engine == "vllm":
        if args.prompts_limit:
            os.environ["PROMPTS_LIMIT"] = str(args.prompts_limit)
        # 跨设备可比: 统一 token id 钳制输入
        os.environ["PROMPTS_AS_TOKENS"] = "1"
        outs = [run_once(args.device, "reference", 10_000 + i, GOLDEN_DIR)
                for i in range(n_snap)]
        all_outputs = [o["output_token_ids"] for o in outs]
        prompts = outs[0]["prompts"]
        input_mode = outs[0].get("input_mode", "text")
    else:
        all_outputs = [transformers_run(profile, args.prompts_limit)
                       for _ in range(n_snap)]
        prompts = load_prompts(args.prompts_limit)
        input_mode = "tokens-clamped-vocab-transformers"
        outs = [{"output_token_ids": o} for o in all_outputs]

    n_prompts = len(outs[0]["output_token_ids"])
    consensus, stable = [], []
    for pi in range(n_prompts):
        seqs = [o["output_token_ids"][pi] for o in outs]
        prefix = consensus_prefix(seqs, args.prefix)
        consensus.append(prefix)
        stable.append(len(prefix))

    golden = {
        "device": args.device,
        "mode": "reference",
        "engine": engine,
        "input_mode": input_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine_args": profile.engine_args(),
        "prompts": prompts,
        "prompts_sha256": hashlib.sha256(
            "\n".join(prompts).encode()).hexdigest()[:16],
        "snapshots": all_outputs,
        "consensus_prefix": consensus,
        "stable_prefix_lens": stable,
        "notes": [
            "共识前缀 = N 次独立运行逐位置多数票(要求全员一致)的最长前缀",
            "比对断言建议 prefix_len = min(stable_prefix_lens) 与 8 取小",
            "本栈已知: 跨进程 16+ token 偶发非确定性(见 drift_study)",
            f"引擎: {engine}" + (" (HuggingFace CPU 权威参考)" if engine == "transformers" else ""),
        ],
    }
    out_path = GOLDEN_DIR / f"{args.device}_golden.json"
    out_path.write_text(json.dumps(golden, indent=2))

    # 清理临时 drift 快照文件
    for i in range(n_snap):
        p = GOLDEN_DIR / f"drift_reference_{10_000 + i:05d}.json"
        if p.exists():
            p.unlink()

    print(f"\n黄金输出已保存: {out_path}")
    print(f"  engine        : {engine}")
    print(f"  snapshots     : {n_snap}")
    print(f"  prompts       : {len(prompts)}")
    print(f"  稳定前缀长度  : {stable}")
    print(f"  建议断言前缀  : {min(min(stable), 8)}")


def load_prompts(limit: int) -> list[str]:
    p = ROOT / "tests" / "framework_level" / "inputs" / "prompts.txt"
    prompts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")]
    return prompts[:limit] if limit > 0 else prompts


def transformers_run(profile, prompts_limit: int) -> list[list[int]]:
    """HuggingFace transformers CPU 推理（权威语义参考，确定性）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(prompts_limit)
    fw = profile.framework
    model_path = fw["model_path"]
    seed = fw.get("seed", 0)

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_path)
    vocab = json.loads(
        (Path(model_path) / "config.json").read_text())["vocab_size"]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16)
    model.eval()

    outputs = []
    for p in prompts:
        from common.inputs import tokenize_clamped
        ids = tokenize_clamped(tok, p, vocab)
        enc = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=32, do_sample=False,
                pad_token_id=tok.eos_token_id)
        in_len = enc["input_ids"].shape[1]
        outputs.append(out[0][in_len:].tolist())
        print(f"  [cpu] '{p[:30]}...' -> {len(outputs[-1])} tokens", flush=True)
    del model
    return outputs


if __name__ == "__main__":
    main()
