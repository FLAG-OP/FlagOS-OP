"""Compare two real-vLLM runner reports and require generation-time hits."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("injected", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.baseline.read_text())
    inj = json.loads(args.injected.read_text())
    assert base["mode"] == "baseline" and inj["mode"] == "injected"
    for field in ("model_sha256", "sampling", "engine", "profile", "vllm"):
        assert base[field] == inj[field], field
    assert base["output_token_ids"] == inj["output_token_ids"]
    assert len(inj["output_token_ids"]) == 2
    assert all(len(seq) == 8 for seq in inj["output_token_ids"])
    assert inj["generation_scatter_calls"] > 0
    assert any("apply_top_k_top_p" in frame for frame in inj["first_call_stack"])
    result = {"ok": True, "profile": inj["profile"], "consumer": "vllm.LLM.generate",
              "level": "C2", "output_match": True,
              "generation_scatter_calls": inj["generation_scatter_calls"],
              "overload": "scatter_.src", "model_kind": inj["model_kind"],
              "baseline": args.baseline.name, "injected": args.injected.name,
              "model_sha256": inj["model_sha256"],
              "scope": "P800 gems5, TP=1 eager single-process engine, random Qwen3; top-k/top-p sampling"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
