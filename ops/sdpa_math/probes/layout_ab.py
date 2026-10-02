"""同进程: k/v 布局 (contiguous vs transposed 视图) 对 native/ours 的 decode 影响."""
import sys, time, statistics, json
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import torch, torch_npu
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

# q 永远是 transposed 视图 (e2e 口径)
q = torch.randn(1,1,8,64,device=DEV,dtype=DT).view(1,1,8,64).transpose(1,2)
# k/v: contiguous = torch.cat 产物 (e2e 口径); transposed = randn(1,S,8,64).transpose
k_ctg = torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2).contiguous()
v_ctg = torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2).contiguous()
k_trs = torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)
v_trs = torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)
print("k_ctg stride", tuple(k_ctg.stride()), "k_trs stride", tuple(k_trs.stride()))

def measure():
    out={}
    out["ctg"]=lat(lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k_ctg,v_ctg,None,0.0,False,None))
    out["trs"]=lat(lambda: torch.ops.aten._scaled_dot_product_attention_math(q,k_trs,v_trs,None,0.0,False,None))
    return out

res={"native": measure()}
from register import register_a1
LIBS = register_a1("AutogradPrivateUse1", None, "triton")
res["ours"]=measure()
print(json.dumps(res))
