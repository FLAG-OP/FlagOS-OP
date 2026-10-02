"""同进程 A/B: 先测 native (未注册), 再注册测 ours, 最后 fsdpa —— 消除跨进程方差."""
import sys, time, statistics, json
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import torch, torch_npu, torch.nn.functional as F
DEV, DT = "npu:0", torch.float16

def lat(fn, reps=50, rounds=9):
    for _ in range(30): fn()
    torch.npu.synchronize(); out=[]
    for _ in range(rounds):
        t0=time.perf_counter()
        for _ in range(reps):
            fn(); torch.npu.synchronize()
        out.append((time.perf_counter()-t0)/reps*1e6)
    return round(statistics.median(out),1)

def burst(fn, n=300, rounds=5):
    for _ in range(50): fn()
    torch.npu.synchronize(); out=[]
    for _ in range(rounds):
        t0=time.perf_counter()
        for _ in range(n): fn()
        torch.npu.synchronize()
        out.append((time.perf_counter()-t0)/n*1e6)
    return round(statistics.median(out),1)

mk = lambda s: torch.randn(1,s,8,64,device=DEV,dtype=DT).transpose(1,2)
cases = {"decode": (mk(1), mk(575), False),
         "prefill": (mk(1024), mk(1024), True)}

def measure():
    out={}
    for name,(q,k,c) in cases.items():
        op = lambda q=q,k=k,c=c: torch.ops.aten._scaled_dot_product_attention_math(q,k,k,None,0.0,c,None)
        out[name+"_lat"]=lat(op); out[name+"_burst"]=burst(op)
    return out

res={"native": measure()}
from register import register_a1
LIBS = register_a1("AutogradPrivateUse1", None, "triton")
res["ours"]=measure()

def measure_fsdpa():
    out={}
    for name,(q,k,c) in cases.items():
        op = lambda q=q,k=k,c=c: F.scaled_dot_product_attention(q,k,k,is_causal=c)
        out[name+"_lat"]=lat(op); out[name+"_burst"]=burst(op)
    return out
res["fsdpa"]=measure_fsdpa()
print(json.dumps(res))
