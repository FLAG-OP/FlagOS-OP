"""decode 差距分解 (正确保留 lib): direct / reg_nograd / reg_grad / native, host 发射时间."""
import sys, time, statistics, json
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import torch, torch_npu
DEV, DT = "npu:0", torch.float16
q=torch.randn(1,1,8,64,device=DEV,dtype=DT).transpose(1,2)
k=torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)
v=torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)

from register import register_a1, _IMPLS
_impl = _IMPLS["triton"]
LIBS = register_a1("AutogradPrivateUse1", None, "triton")

def lat(fn, reps=40, rounds=7, sync=True):
    for _ in range(20): fn()
    torch.npu.synchronize(); out=[]
    for _ in range(rounds):
        t0=time.perf_counter()
        for _ in range(reps):
            fn()
            if sync: torch.npu.synchronize()
        out.append((time.perf_counter()-t0)/reps*1e6)
    return round(statistics.median(out),1)

def host(fn, n=300):
    for _ in range(50): fn()
    torch.npu.synchronize()
    t0=time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return round((time.perf_counter()-t0)/n*1e6,1)

op = lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,False,None)
res={}
# grad 开 (e2e prefill 默认图? no_grad 下 e2e 是关的)
res["reg_grad_lat"]=lat(op)
res["reg_grad_host"]=host(op)
# no_grad (e2e 口径)
with torch.no_grad():
    res["reg_nograd_lat"]=lat(op)
    res["reg_nograd_host"]=host(op)
    res["direct_lat"]=lat(lambda: _impl(q,k,v,None,0.0,False,None,scale=None,enable_gqa=False))
    res["direct_host"]=host(lambda: _impl(q,k,v,None,0.0,False,None,scale=None,enable_gqa=False))
    res["direct_thr"]=lat(lambda: _impl(q,k,v,None,0.0,False,None,scale=None,enable_gqa=False), sync=False)
    res["reg_nograd_thr"]=lat(op, sync=False)
print(json.dumps(res))
