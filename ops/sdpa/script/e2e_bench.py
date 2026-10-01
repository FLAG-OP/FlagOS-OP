# 端到端基准（子进程隔离版）: mini-LLM 全原生 vs 生产链 vs 纯Triton
# 用法: python3 e2e_bench.py            # 跑全部三链
#       python3 e2e_bench.py --phase A  # 单链（内部复用）
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SDPA_DIR = "/root/sdpatten-op"
EMB_DIR = "/root/FlagOS-OP/ops/embedding"
DEV = "npu:0"

MODEL_SRC = '''
import sys, time
sys.path.insert(0, "/root/sdpatten-op")   # sdpa 的 kernel/ 优先（phase 内单算子语义）

PHASE = sys.argv[1]
import json
import torch, torch_npu
import torch.nn as nn

class Attn(nn.Module):
    def __init__(self, d, nh, nkv):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, d // nh
        self.wq = nn.Linear(d, nh*self.hd, bias=False)
        self.wk = nn.Linear(d, nkv*self.hd, bias=False)
        self.wv = nn.Linear(d, nkv*self.hd, bias=False)
        self.wo = nn.Linear(nh*self.hd, d, bias=False)
    def forward(self, x, kv=None):
        B, S, _ = x.shape
        q = self.wq(x).view(B,S,self.nh,self.hd).transpose(1,2)
        k = self.wk(x).view(B,S,self.nkv,self.hd).transpose(1,2)
        v = self.wv(x).view(B,S,self.nkv,self.hd).transpose(1,2)
        if kv is not None:
            k = torch.cat([kv[0], k], 2); v = torch.cat([kv[1], v], 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=(S==k.shape[2]), enable_gqa=True)
        return self.wo(o.transpose(1,2).reshape(B,S,-1)), ((k,v) if kv is not None else None)

class Block(nn.Module):
    def __init__(s, d, nh, nkv, h):
        super().__init__()
        s.attn = Attn(d,nh,nkv); s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d,h), nn.GELU(), nn.Linear(h,d))
    def forward(s, x, kv=None):
        a, nc = s.attn(s.ln1(x), kv); x = x + a
        return x + s.mlp(s.ln2(x)), nc

class MiniLLM(nn.Module):
    def __init__(s, vocab=32768, d=512, layers=4, nh=8, nkv=2, h=2048):
        super().__init__()
        s.tok = nn.Embedding(vocab, d)
        s.blocks = nn.ModuleList(Block(d,nh,nkv,h) for _ in range(layers))
        s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, vocab, bias=False)
    def forward(s, ids, kvs=None):
        h = s.tok(ids)
        ncs = []
        for i, b in enumerate(s.blocks):
            h, nc = b(h, kvs[i] if kvs else None); ncs.append(nc)
        return s.head(s.ln(h)), ncs

def bench(fn, w=5, it=20):
    for _ in range(w): fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.npu.synchronize()
    return (time.perf_counter()-t0)/it*1000

torch.manual_seed(2026)
model = MiniLLM().to("npu:0").to(torch.float16).eval()
g = torch.Generator().manual_seed(7)

if PHASE == "B":
    import os
    os.environ["SDPA_DISPATCH_MODE"] = "auto"
    # embedding 先加载并占住顶层名 "register"（其内部 import register 指自身）
    sys.path.insert(0, "/root/FlagOS-OP/ops/embedding")
    import register as emb_register
    emb_register.register_a1("AutogradPrivateUse1", platform="ascend910")
    # sdpa 后加载: auto_dispatch 内部已按文件路径锚定（不受名字占用影响）
    from kernel.auto_dispatch import install_patch
    install_patch()
elif PHASE == "C":
    import os
    os.environ["SDPA_DISPATCH_MODE"] = "triton"
    from kernel.auto_dispatch import install_patch
    install_patch()

CFGS = [("prefill_1k_b8",8,1024),("prefill_2k_b4",4,2048),
        ("prefill_4k_b2",2,4096),("decode_b32",32,1)]
out = {}
with torch.no_grad():
    for tag, B, S in CFGS:
        ids = torch.randint(0,32768,(B,S),generator=g).to("npu:0")
        if tag.startswith("decode"):
            prompt = torch.randint(0,32768,(B,511),generator=g).to("npu:0")
            _, caches = model(prompt)
            t = bench(lambda: model(ids, caches))
        else:
            t = bench(lambda: model(ids))
        out[tag] = round(t,3)
        print(f"{PHASE} {tag:16s} {t:8.2f} ms", flush=True)
print("RESULT", json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None, help="A|B|C（内部用）")
    args = ap.parse_args()

    if args.phase:  # 子进程模式由 runner 写临时文件
        pass

    runner = Path("/tmp/e2e_model_runner.py")
    runner.write_text(MODEL_SRC)

    results = {}
    for phase in ("A", "B", "C"):
        r = subprocess.run([sys.executable, str(runner), phase],
                           capture_output=True, text=True, timeout=1200)
        line = [ln for ln in r.stdout.splitlines()
                if ln.startswith("RESULT")]
        if not line:
            print(f"[{phase}] FAIL:\n", r.stdout[-600:], r.stderr[-400:])
            results[phase] = None
            continue
        import json as _j
        results[phase] = _j.loads(line[0].split(" ", 1)[1])

    print()
    print("=" * 68)
    print(f"{'场景':16s} {'A 原生':>9s} {'B 生产链':>9s} {'B/A':>7s} "
          f"{'C 纯Triton':>10s} {'C/A':>7s}")
    print("-" * 68)
    for tag in ("prefill_1k_b8", "prefill_2k_b4", "prefill_4k_b2",
                "decode_b32"):
        a = results["A"][tag] if results["A"] else float("nan")
        b = results["B"][tag] if results["B"] else float("nan")
        c = results["C"][tag] if results["C"] else float("nan")
        print(f"{tag:16s} {a:8.2f}ms {b:8.2f}ms {b/a:6.2f}x "
              f"{c:9.2f}ms {c/a:6.2f}x")
    Path("/root/sdpatten-op/reports/e2e_mini_llm.json").write_text(
        json.dumps(results, indent=2))
    print("\nsaved -> reports/e2e_mini_llm.json")


if __name__ == "__main__":
    main()
