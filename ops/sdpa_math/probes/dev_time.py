"""同进程 device 时间 (event) + host 时间: native vs ours direct vs ours reg."""
import sys, time, statistics, json
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import torch, torch_npu
DEV, DT = "npu:0", torch.float16
q=torch.randn(1,1,8,64,device=DEV,dtype=DT).transpose(1,2)
k=torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)
v=torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)

def devtime(fn, n=40):
    for _ in range(30): fn()
    torch.npu.synchronize(); out=[]
    for _ in range(n):
        s=torch.npu.Event(enable_timing=True); e=torch.npu.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.npu.synchronize()
        out.append(s.elapsed_time(e)*1e3)
    return round(statistics.median(out),1)

def host(fn, n=300):
    for _ in range(50): fn()
    torch.npu.synchronize()
    t0=time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return round((time.perf_counter()-t0)/n*1e6,1)

res={}
res["native_dev"]=devtime(lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,False,None))
res["native_host"]=host(lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,False,None))
from register import register_a1, _IMPLS
_impl=_IMPLS["triton"]
with torch.no_grad():
    res["direct_dev"]=devtime(lambda: _impl(q,k,v,None,0.0,False,None,scale=None,enable_gqa=False))
    res["direct_host"]=host(lambda: _impl(q,k,v,None,0.0,False,None,scale=None,enable_gqa=False))
LIBS=register_a1("AutogradPrivateUse1", None, "triton")
res["reg_dev"]=devtime(lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,False,None))
res["reg_host"]=host(lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,False,None))
print(json.dumps(res))
