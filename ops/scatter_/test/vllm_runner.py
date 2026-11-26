"""Real vLLM generation, baseline or A1 injection, using a local tiny model.

Run each mode in a fresh process in the target container. Single-process v1
engine keeps the measured registration/counters in the model worker process.
"""
import argparse
import json
import hashlib
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["USE_FLAGGEMS"] = "0"
os.environ.setdefault("TASK_QUEUE_ENABLE", "0")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("prepare", "baseline", "injected"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-profile", default="ascend910")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import torch
    if args.mode == "prepare":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(20261126)
        config = Qwen3Config(vocab_size=1024, hidden_size=128,
                            intermediate_size=256, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1,
                            head_dim=64, max_position_embeddings=256,
                            bos_token_id=1, eos_token_id=2, pad_token_id=0,
                            tie_word_embeddings=False)
        model = Qwen3ForCausalLM(config).to(torch.bfloat16)
        model.save_pretrained(args.model)
        print("Saved deterministic random-weight Qwen3 model:", args.model)
        return

    import torch_npu  # noqa: F401
    from common.device import load_profile
    from vllm import LLM, SamplingParams
    profile = load_profile(args.device_profile)
    llm = LLM(model=args.model, skip_tokenizer_init=True, dtype="bfloat16",
              enforce_eager=True, max_model_len=128, max_num_seqs=2,
              max_num_batched_tokens=128, tensor_parallel_size=1,
              seed=20261126, enable_prefix_caching=False,
              disable_log_stats=True)

    class Counter(dict):
        first_stack = None

        def __setitem__(self, key, value):
            if key == "n" and self.first_stack is None:
                self.first_stack = traceback.format_stack(limit=20)
            super().__setitem__(key, value)

    counter = Counter(n=0)
    lib = None
    if args.mode == "injected":
        from ops.scatter_ import register_a1
        lib = register_a1(profile.dispatch_key, counter=counter,
                          platform=profile.vendor)
    prompts = [{"prompt_token_ids": [1, 17, 23, 42]},
               {"prompt_token_ids": [1, 73, 8, 16, 39]}]
    params = SamplingParams(max_tokens=8, temperature=0.8, top_k=32,
                            top_p=0.9, seed=1234, ignore_eos=True,
                            detokenize=False)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    result = {"mode": args.mode, "model": args.model,
              "profile": profile.name,
              "model_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(args.model).iterdir())
                               if p.suffix in (".json", ".safetensors")},
              "model_kind": "deterministic random-weight Qwen3, not pretrained",
              "consumer": "vllm.LLM.generate", "vllm": __import__("vllm").__version__,
              "output_token_ids": [list(o.outputs[0].token_ids) for o in outputs],
              "generation_scatter_calls": counter["n"],
              "first_call_stack": counter.first_stack,
              "sampling": {"temperature": 0.8, "top_k": 32, "top_p": 0.9,
                           "seed": 1234, "max_tokens": 8},
              "engine": {"enforce_eager": True, "multiprocessing": False,
                         "tensor_parallel_size": 1}}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if lib is not None:
        lib._destroy()
    llm.llm_engine.engine_core.shutdown()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
