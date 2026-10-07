"""进程内 A/B: 当前实现 vs 强制 contiguous (还原旧行为)."""
import sys, time, importlib
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "kernel"))
import torch, torch_npu
tl_mod = importlib.import_module("triton_level")
DEV="npu:0"; DT=torch.bfloat16

def bench(fn, warm=150, reps=600, rounds=9):
    for _ in range(warm): fn()
    torch.npu.synchronize(); out=[]
    for _ in range(rounds):
        t0=time.perf_counter()
        for _ in range(reps): fn()
        torch.npu.synchronize(); out.append((time.perf_counter()-t0)/reps*1e6)
    return min(out), sorted(out)[len(out)//2]

def mk(B,H,D,skv,sq):
    q = torch.randn(B,sq,H,D,device=DEV,dtype=DT).transpose(1,2)
    k = torch.randn(B,skv,H,D,device=DEV,dtype=DT).transpose(1,2)
    v = torch.randn(B,skv,H,D,device=DEV,dtype=DT).transpose(1,2)
    return q,k,v

cases = {
  "decode(1x575)":  mk(1,8,64,575,1),
  "prefill(1024)":  mk(1,8,64,1024,1024),
}
orig_d1 = tl_mod._d1
for name,(q,k,v) in cases.items():
    causal = name.startswith("prefill")
    new = bench(lambda: tl_mod.sdpa_math_triton(q,k,v,None,0.0,causal,None))
    tl_mod._d1 = lambda x: x.contiguous()   # 还原旧行为
    old = bench(lambda: tl_mod.sdpa_math_triton(q,k,v,None,0.0,causal,None))
    tl_mod._d1 = orig_d1
    print(f"{name}: NEW min={new[0]:.1f}/med={new[1]:.1f}µs  OLD min={old[0]:.1f}/med={old[1]:.1f}µs  节省={old[0]-new[0]:.1f}µs")
