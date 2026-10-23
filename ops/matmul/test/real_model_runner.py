#!/usr/bin/env python3
"""Generate a few tokens with vLLM; the optional A1 bridge runs at startup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--marker", type=Path)
    ap.add_argument("--tensor-parallel-size", type=int, required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model, tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128,
        gpu_memory_utilization=0.1, dtype="float16", seed=20260930,
        disable_log_stats=True, trust_remote_code=False,
    )
    if args.marker is not None:
        args.marker.touch()
    outputs = llm.generate(
        ["the future of AI is"],
        SamplingParams(max_tokens=4, temperature=0.0, detokenize=False),
    )
    ids = [list(output.outputs[0].token_ids) for output in outputs]
    args.out.write_text(json.dumps({"output_token_ids": ids}) + "\n")
    print(f"model output token ids: {ids}", flush=True)


if __name__ == "__main__":
    main()
