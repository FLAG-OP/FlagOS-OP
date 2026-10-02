#!/usr/bin/env python3
"""FlagOS end-to-end probe for sdpa_math on Kunlunxin P800.

The workload is a small Llama-style model whose attention calls the private
math operator directly and consumes the returned probability map through an
entropy regularizer.  Both native and A1 phases run inside the FlagOS stack:
FlagGems supplies the surrounding GELU, while the target attention op remains
aten::_scaled_dot_product_attention_math.

The two phases are run in separate subprocesses because torch.library A1
registrations cannot be undone in-process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D_MODEL = 128
N_LAYER = 4
N_HEAD = 8
N_KV_HEAD = 2
HEAD_DIM = D_MODEL // N_HEAD
N_KV_DIM = N_KV_HEAD * HEAD_DIM
VOCAB = 1024
FFN = 256


def _build_model(device: str, dtype):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MiniProbLLM(nn.Module):
        """Four-block decoder with GQA and an attention-entropy consumer."""

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB, D_MODEL)
            self.layers = nn.ModuleList()
            for _ in range(N_LAYER):
                self.layers.append(nn.ModuleDict({
                    "norm1": nn.LayerNorm(D_MODEL, elementwise_affine=False),
                    "q": nn.Linear(D_MODEL, D_MODEL, bias=False),
                    "k": nn.Linear(D_MODEL, N_KV_DIM, bias=False),
                    "v": nn.Linear(D_MODEL, N_KV_DIM, bias=False),
                    "o": nn.Linear(D_MODEL, D_MODEL, bias=False),
                    "norm2": nn.LayerNorm(D_MODEL, elementwise_affine=False),
                    "f1": nn.Linear(D_MODEL, FFN, bias=False),
                    "f2": nn.Linear(FFN, D_MODEL, bias=False),
                }))
            self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

        def forward(self, ids):
            batch, seq = ids.shape
            x = self.embedding(ids)
            entropy = x.new_zeros(())
            for layer in self.layers:
                h = layer["norm1"](x)
                q = (layer["q"](h).view(batch, seq, N_HEAD, HEAD_DIM)
                     .transpose(1, 2))
                k = (layer["k"](h).view(batch, seq, N_KV_HEAD, HEAD_DIM)
                     .transpose(1, 2))
                v = (layer["v"](h).view(batch, seq, N_KV_HEAD, HEAD_DIM)
                     .transpose(1, 2))
                ctx, probs = torch.ops.aten._scaled_dot_product_attention_math(
                    q, k, v, None, 0.0, True, None, enable_gqa=True,
                )
                p32 = probs.float().clamp_min(1e-9)
                entropy = entropy - (p32 * p32.log()).sum(dim=-1).mean()
                x = x + layer["o"](
                    ctx.transpose(1, 2).reshape(batch, seq, D_MODEL))
                mlp = layer["f2"](F.gelu(layer["f1"](layer["norm2"](x))))
                x = x + mlp
            return self.head(x), entropy / N_LAYER

    torch.manual_seed(20261002)
    return MiniProbLLM().to(device).to(dtype)


def _consume(output):
    logits, auxiliary = output
    value = logits.reshape(-1)[0].item()
    return value if auxiliary is None else value + auxiliary.item()


def _bench(fn, device: str, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        _consume(fn())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _consume(fn())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def _generate(model, prompt, steps: int = 8):
    import torch

    tokens = [torch.full((prompt.shape[0],), 0, dtype=torch.long,
                         device=prompt.device)]
    tokens.extend(prompt.t())
    with torch.no_grad():
        for _ in range(steps):
            sequence = torch.stack(tokens, dim=1)
            logits, _ = model(sequence)
            tokens.append(logits[:, -1].argmax(dim=-1))
    return torch.stack(tokens[1:], dim=1)


def _run_phase(args, artifact_dir: Path) -> dict:
    import torch
    import torch.nn.functional as F

    # Full FlagGems enablement is not deterministic on the locked P800 image.
    # GELU is stable and is genuinely consumed by this model, so this keeps the
    # surrounding computation inside FlagOS without making the test flaky.
    import flag_gems
    flag_gems.only_enable(include=["gelu"])

    device = args.device
    dtype = getattr(torch, args.dtype)
    model = _build_model(device, dtype).eval()
    counter = {"n": 0}
    libs = None
    if args.phase == "plugin":
        from ops.sdpa_math import register_a1
        libs = register_a1("AutogradCUDA", counter=counter, impl="p800")

    generator = torch.Generator(device="cpu").manual_seed(31)
    workloads = [
        ("forward_b1_s256", 1, 256),
        ("forward_b2_s128", 2, 128),
        ("train_b1_s128", 1, 128),
        ("generate8_b1_s96", 1, 96),
    ]
    result = {
        "phase": args.phase,
        "device": device,
        "dtype": args.dtype,
        "flag_gems": ["gelu"],
        "model": {
            "layers": N_LAYER,
            "d_model": D_MODEL,
            "heads": f"{N_HEAD}/{N_KV_HEAD}",
            "vocab": VOCAB,
            "ffn": FFN,
            "probs_consumer": "attention entropy regularizer",
        },
        "cases": {},
        "interceptions": 0,
    }

    for tag, batch, seq in workloads:
        ids = torch.randint(
            0, VOCAB, (batch, seq), generator=generator).to(device)
        if tag.startswith("forward"):
            def call():
                with torch.no_grad():
                    return model(ids)
            warmup, iters = 10, 30
        elif tag.startswith("train"):
            target = torch.randint(
                0, VOCAB, (batch, seq), generator=generator).to(device)

            def call():
                model.zero_grad(set_to_none=True)
                logits, entropy = model(ids)
                loss = (
                    F.cross_entropy(
                        logits.reshape(-1, VOCAB), target.reshape(-1))
                    - 0.01 * entropy
                )
                loss.backward()
                return logits, loss
            warmup, iters = 5, 15
        else:
            def call():
                generated = _generate(model, ids)
                return generated[:, -1], None
            warmup, iters = 3, 10

        ms = _bench(call, device, warmup, iters)
        result["cases"][tag] = round(ms, 4)
        print(f"{args.phase:6s} {tag:20s} {ms:9.3f} ms", flush=True)

    # Fixed correctness sample.  Save compact tensors so the parent process can
    # compare native and plugin results obtained in isolated subprocesses.
    ids = torch.randint(
        0, VOCAB, (1, 160), generator=generator).to(device)
    with torch.no_grad():
        logits, entropy = model(ids)
    generated = _generate(model, ids[:, :32])

    model.zero_grad(set_to_none=True)
    target = torch.randint(
        0, VOCAB, (1, 160), generator=generator).to(device)
    logits_for_grad, entropy_for_grad = model(ids)
    loss = (F.cross_entropy(
        logits_for_grad.reshape(-1, VOCAB), target.reshape(-1))
        - 0.01 * entropy_for_grad)
    loss.backward()

    torch.save({
        "logits": logits.detach().float().cpu(),
        "entropy": entropy.detach().float().cpu(),
        "generated": generated.cpu(),
        "embedding_grad": model.embedding.weight.grad.detach().float().cpu(),
        "q0_grad": model.layers[0]["q"].weight.grad.detach().float().cpu(),
        "head_grad": model.head.weight.grad.detach().float().cpu(),
    }, artifact_dir / f"{args.phase}.pt")

    result["interceptions"] = counter["n"]
    result["fixed_sample"] = {
        "logits_shape": list(logits.shape),
        "logits_sum": float(logits.float().sum()),
        "entropy": float(entropy.float()),
        "generated": generated[0].tolist(),
    }
    if libs is not None:
        # Keep the registration objects alive through the final checks; the
        # process exits immediately afterward.
        assert len(libs) == 2
    print("RESULT " + json.dumps(result), flush=True)
    return result


def _compare(left: dict, right: dict, artifact_dir: Path) -> dict:
    import torch

    native = torch.load(artifact_dir / "native.pt", weights_only=False)
    plugin = torch.load(artifact_dir / "plugin.pt", weights_only=False)
    logits_diff = (native["logits"] - plugin["logits"]).abs().max().item()
    native_top1 = native["logits"].argmax(dim=-1)
    plugin_top1 = plugin["logits"].argmax(dim=-1)
    top1_rate = (native_top1 == plugin_top1).float().mean().item()
    entropy_diff = abs(
        float(native["entropy"]) - float(plugin["entropy"]))
    sequence_match = torch.equal(native["generated"], plugin["generated"])
    gradients = {}
    for key in ("embedding_grad", "q0_grad", "head_grad"):
        gradients[key] = round(
            (native[key] - plugin[key]).abs().max().item(), 8)

    checks = {
        "logits_linf": logits_diff,
        "top1_rate": round(top1_rate, 6),
        "entropy_abs": entropy_diff,
        "greedy_sequence_match": sequence_match,
        **gradients,
    }
    ok = (
        logits_diff < 0.02
        and top1_rate >= 0.98
        and entropy_diff < 0.01
        and sequence_match
        and all(value < 0.02 for value in gradients.values())
    )
    return {"ok": ok, **checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("native", "plugin"),
                        help="internal subprocess mode")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16"))
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--json-out",
                        default=OP_DIR / "reports/e2e_flagos_p800-kunlunxin.json")
    args = parser.parse_args()

    if args.phase:
        if args.artifact_dir is None:
            parser.error("--artifact-dir is required with --phase")
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        _run_phase(args, args.artifact_dir)
        return 0

    with tempfile.TemporaryDirectory(prefix="sdpa-math-e2e-") as temporary:
        artifact_dir = Path(temporary)
        results = {}
        for phase in ("native", "plugin"):
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--phase", phase,
                "--device", args.device,
                "--dtype", args.dtype,
                "--artifact-dir", str(artifact_dir),
            ]
            process = subprocess.run(
                command, text=True, capture_output=True, timeout=600,
            )
            lines = [line for line in process.stdout.splitlines()
                     if line.startswith("RESULT ")]
            if process.returncode or not lines:
                print(process.stdout)
                print(process.stderr)
                raise RuntimeError(f"{phase} phase failed")
            print("\n".join(
                line for line in process.stdout.splitlines()
                if not line.startswith("RESULT ")))
            results[phase] = json.loads(lines[-1][len("RESULT "):])

        comparison = _compare(results["native"], results["plugin"], artifact_dir)
        summary = {}
        for case in results["native"]["cases"]:
            native_ms = results["native"]["cases"][case]
            plugin_ms = results["plugin"]["cases"][case]
            summary[case] = {
                "native_ms": native_ms,
                "plugin_ms": plugin_ms,
                "speedup_native_over_plugin": round(native_ms / plugin_ms, 3),
            }

        report = {
            "ok": comparison["ok"],
            "framework": "FlagOS/FlagGems GELU + sdpa_math A1",
            "comparison": comparison,
            "performance": summary,
            "native": results["native"],
            "plugin": results["plugin"],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")

        print("\n" + "=" * 88)
        print(f"FlagOS E2E: {args.device} {args.dtype}; "
              f"model={N_LAYER}L D={D_MODEL} H={N_HEAD}/{N_KV_HEAD}")
        print("=" * 88)
        for case, row in summary.items():
            print(f"{case:20s} native={row['native_ms']:8.3f}ms "
                  f"plugin={row['plugin_ms']:8.3f}ms "
                  f"speedup={row['speedup_native_over_plugin']:6.2f}x")
        print("-" * 88)
        print(json.dumps(comparison, indent=2))
        print(f"saved -> {args.json_out}")
        if not report["ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
