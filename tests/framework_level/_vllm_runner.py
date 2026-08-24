#!/usr/bin/env python3
"""框架级测试子进程 runner（设备/引擎参数全部来自 profile）。

用法: _vllm_runner.py <out.json> --device <name> [--with-plugin] [--route a1|a2|b]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--device", required=True)
    ap.add_argument("--route", default="b", choices=["a1", "a2", "b"])
    ap.add_argument("--with-plugin", action="store_true")
    args = ap.parse_args()

    from common.device import load_profile
    profile = load_profile(args.device)

    if args.with_plugin:
        if args.route == "a1":
            # aten 注册由 injection/sitecustomize.py 在每个子进程内完成
            # （vLLM v1 前向跑在 EngineCore 子进程，主进程注册不传播）。
            # 这里只需迫使 silu_and_mul dispatch 走 reference → F.silu
            os.environ["VLLM_FL_PER_OP"] = "silu_and_mul=reference"
        elif args.route == "a2":
            os.environ["VLLM_FL_PER_OP"] = \
                "silu_and_mul=vendor:triton-template|reference"
        elif args.route == "b":
            os.environ.setdefault("AUDIT_DELEGATE", "vendor")
            os.environ["VLLM_FL_PER_OP"] = "silu_and_mul=vendor:audit|reference"

    from vllm import LLM, SamplingParams

    llm = LLM(**profile.engine_args())

    # 输入外置: inputs/prompts.txt（每行一个 prompt; 可用 PROMPTS_FILE 覆盖，
    # PROMPTS_LIMIT 限制条数便于快速冒烟/长时间计时）
    prompts_file = Path(os.environ.get(
        "PROMPTS_FILE",
        Path(__file__).parent / "inputs" / "prompts.txt",
    ))
    prompts = [
        line.strip() for line in prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    limit = int(os.environ.get("PROMPTS_LIMIT", "0"))
    if limit > 0:
        prompts = prompts[:limit]

    # 跨设备可比模式: token id 输入 + 词表钳制（与 CPU transformers
    # 黄金生成使用完全相同的输入序列; 规避 vendor 栈 embedding 静默越界）
    input_mode = "text"
    if os.environ.get("PROMPTS_AS_TOKENS") == "1":
        from transformers import AutoTokenizer
        from vllm import TokensPrompt
        from common.inputs import tokenize_clamped

        model_path = profile.framework["model_path"]
        vocab = json.loads(
            (Path(model_path) / "config.json").read_text())["vocab_size"]
        tok = AutoTokenizer.from_pretrained(model_path)
        gen_inputs = [TokensPrompt(prompt_token_ids=tokenize_clamped(tok, p, vocab))
                      for p in prompts]
        input_mode = f"tokens-clamped-vocab-{vocab}"
    else:
        gen_inputs = prompts
    params = SamplingParams(max_tokens=32, temperature=0.0, seed=20260823,
                            detokenize=False)
    outs = llm.generate(gen_inputs, params)
    result = {
        "route": args.route,
        "device": profile.name,
        "with_plugin": args.with_plugin,
        "prompt_token_ids": [list(o.prompt_token_ids) for o in outs],
        "output_token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "prompts": prompts,
        "input_mode": input_mode,
    }
    Path(args.out_json).write_text(json.dumps(result))
    print(f"runner done route={args.route} with_plugin={args.with_plugin}")


if __name__ == "__main__":
    main()
