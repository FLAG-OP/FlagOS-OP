"""同进程 e2e 上下文 A/B: 在 MiniDecoder 模型上下文内先测 native, 再注册测 ours.
消除跨进程漂移. 每步 st() 双侧 sync. 可选 SKIP_ENTROPY=1 跳过熵消费."""
import sys, time, statistics, json, os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import torch, torch.nn.functional as F, torch_npu
DEV, DT = "npu:0", torch.float16
N_LAYERS, D, H, HEAD, V = 4, 512, 8, 64, 512
PREFILL_S, CTX0, STEPS = 1024, 512, 64
SKIP_ENTROPY = os.environ.get("SKIP_ENTROPY") == "1"

def do_attn(q,k,v,causal):
    return torch.ops.aten._scaled_dot_product_attention_math(q,k,v,None,0.0,causal,None)
def entropy(p): return -(p*p.clamp_min(1e-9).log()).sum(-1).mean()

def st(fn):
    torch.npu.synchronize(); t0=time.perf_counter(); r=fn(); torch.npu.synchronize()
    return r, (time.perf_counter()-t0)*1e6

g = torch.Generator("cpu").manual_seed(0)
emb=(torch.randn(V,D,generator=g)*0.1).to(DEV).to(DT)
mk=lambda a,b:(torch.randn(a,b,generator=g)*0.05).to(DEV).to(DT)
layers=[dict(Wq=mk(D,D),Wk=mk(D,D),Wv=mk(D,D),Wo=mk(D,D),W1=mk(D,4*D),W2=mk(4*D,D)) for _ in range(N_LAYERS)]
Wout=mk(D,V)

def measure():
    S={"proj":[],"cat":[],"attn":[],"entropy":[],"rest":[]}
    with torch.no_grad():
        ptok=torch.zeros(1,PREFILL_S,dtype=torch.long,device=DEV)
        x=emb[ptok]; caches=[]
        for L in layers:
            k=(x@L["Wk"]).view(1,PREFILL_S,H,HEAD).transpose(1,2)
            v=(x@L["Wv"]).view(1,PREFILL_S,H,HEAD).transpose(1,2); caches.append([k,v])
            q=(x@L["Wq"]).view(1,PREFILL_S,H,HEAD).transpose(1,2)
            (ctx,p),_=st(lambda q=q,k=k,v=v: do_attn(q,k,v,True))
            if p is not None and not SKIP_ENTROPY: _,_=st(lambda p=p: entropy(p))
            x=x+(ctx.transpose(1,2).reshape(1,PREFILL_S,D)@L["Wo"])
            x=x+F.gelu(x@L["W1"])@L["W2"]
        logits=(x@Wout)@emb.t()
        tok=torch.tensor([[int(logits[0,-1].argmax().item())]],device=DEV)
        x1=emb[tok]
        for step in range(STEPS):
            for li,L in enumerate(layers):
                k,v=caches[li]
                (q,kt,vt),dt=st(lambda L=L: ((x1@L["Wq"]).view(1,1,H,HEAD).transpose(1,2),
                                             (x1@L["Wk"]).view(1,1,H,HEAD).transpose(1,2),
                                             (x1@L["Wv"]).view(1,1,H,HEAD).transpose(1,2)))
                S["proj"].append(dt)
                (kk,vv),dt=st(lambda k=k,kt=kt,v=v,vt=vt: (torch.cat([k,kt],dim=2),torch.cat([v,vt],dim=2)))
                S["cat"].append(dt)
                (ctx,p),dt=st(lambda q=q,kk=kk,vv=vv: do_attn(q,kk,vv,False))
                S["attn"].append(dt)
                if p is not None and not SKIP_ENTROPY:
                    _,dt=st(lambda p=p: entropy(p)); S["entropy"].append(dt)
                x1,dt=st(lambda x1=x1,ctx=ctx,L=L: x1+(ctx.transpose(1,2).reshape(1,1,D)@L["Wo"]))
                S["rest"].append(dt)
                x1,dt=st(lambda x1=x1,L=L: x1+F.gelu(x1@L["W1"])@L["W2"])
                S["rest"].append(dt)
            logits,dt=st(lambda: (x1@Wout)@emb.t())
            nxt=int(logits[0,-1].argmax().item())
            tok=torch.tensor([[nxt]],device=DEV); x1=emb[tok]
    return {k: round(statistics.median(v),1) for k,v in S.items()}

res={"native": measure()}
from register import register_a1
LIBS=register_a1("AutogradPrivateUse1", {"n":0}, "triton")
res["ours"]=measure()
print(json.dumps(res))
