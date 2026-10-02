"""sdpa_math_triton host 阶段计时 (decode 形状): validate/alloc/launch 各占多少."""
import sys, time, statistics, json, importlib
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "kernel"))
import torch, torch_npu
DEV, DT = "npu:0", torch.float16
q=torch.randn(1,1,8,64,device=DEV,dtype=DT).transpose(1,2)
k=torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)
v=torch.randn(1,575,8,64,device=DEV,dtype=DT).transpose(1,2)
import triton_level as TL

def host(fn, n=300, warm=50):
    for _ in range(warm): fn()
    torch.npu.synchronize()
    t0=time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return round((time.perf_counter()-t0)/n*1e6,1)

res={}
res["full"]=host(lambda: TL.sdpa_math_triton(q,k,v,None,0.0,False,None))
# 阶段: 只跑 _validate
res["validate"]=host(lambda: TL._validate(q,k,v,None,False,False))
# 只跑 _d1 x3
res["d1"]=host(lambda: (TL._d1(q),TL._d1(k),TL._d1(v)))
# 只跑 empty+view
n_p=1*8*1*575; D=64
res["alloc"]=host(lambda: (lambda b:(b[:n_p].view(1,8,1,575), b[n_p:].view(1,8,1,64)))(torch.empty(n_p+1*8*1*D,dtype=DT,device=DEV)))
# 只跑 kernel launch (手动复刻调用)
from flag_gems.runtime import torch_device_fn
BLOCK_M,BLOCK_N=TL._blocks(64)
import math, triton
sm_scale=1.0/math.sqrt(64)
probs=torch.empty(n_p,dtype=DT,device=DEV).view(1,8,1,575)
out=torch.empty(1*8*1*64,dtype=DT,device=DEV).view(1,8,1,64)
grid=(triton.cdiv(1,BLOCK_M), 1*8)
def launch():
    with torch_device_fn.device(q.device):
        TL._probs_kernel[grid](
            q,k,probs,v,out,1,575,sm_scale,8,1,
            q.stride(0),q.stride(1),q.stride(2),
            k.stride(0),k.stride(1),k.stride(2),
            probs.stride(0),probs.stride(1),probs.stride(2),probs.stride(3),
            v.stride(0),v.stride(1),v.stride(2),
            out.stride(0),out.stride(1),out.stride(2),
            q,q,0,0,0,0,
            HAS_BMASK=False,HAS_FMASK=False,IS_CAUSAL=False,HAS_PV=False,
            BLOCK_M=BLOCK_M,BLOCK_N=BLOCK_N,D_POW2=64,D_ACTUAL=64,INPUT_FP32=False,num_warps=4)
res["launch_only"]=host(launch)
# _use_fused
res["use_fused"]=host(lambda: TL._use_fused(1,575,64,False,0.0))
# torch_device_fn.device enter/exit
_dc = torch_device_fn.device(q.device)
def _devctx():
    with _dc: pass
res["dev_ctx"]=host(_devctx)
print(json.dumps(res))
