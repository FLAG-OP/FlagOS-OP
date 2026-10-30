# A1 多算子注入: 在单个进程（含 vLLM EngineCore 子进程）内注册 FlagOS-OP
# 的多个 aten 算子。
#
# 背景: ops/<op>/register.py 的实现用 `from kernel.torch_level import ...`
# 这一「裸 kernel 包名」导入；当多个算子同时注册时，各自的 ops/<op>/kernel/
# 同名包会互相覆盖（先被导入者胜出），导致 A 算子执行到 B 算子的 kernel。
# 本模块为每个算子隔离导入其 kernel 模块，并在该算子的 aten 包装函数调用
# 期间把 sys.modules['kernel*'] 切回它自己的包，从而多算子可同时注册、
# 互不串味（详见 tests/framework_level/test_a1_aten_ops.py）。
#
# 由 injection/sitecustomize.py 在 FLAGOS_A1_ATEN_INJECT=1 时调用。
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]

# 可在真实 vLLM 进程内安全注册的算子（实测）。
# 排除项与原因:
#   empty / detach_ / result_type —— 设计上不可 A1 拦截（见各 op 的 register.py）。
#   detach —— 可独立注册，但在真实 vLLM 建模时注册会破坏
#             torch.Tensor._make_subclass(nn.Parameter 构造)，报
#             "raw Tensor object is already associated to a python object
#             of type Tensor"。与 detach_ 类似，属于进程内不可共存的算子。
_DEFAULT_OPS = [
    "type_as", "clone", "contiguous", "copy_", "dropout",
    "empty_like", "empty_strided", "item", "_local_scalar_dense",
]

# register.py 内无 env 计数文件约定的算子: 由本模块 _FileCounter 落盘，
# 文件环境变量与 tests/framework_level/_aten_harness.py 的 COUNT_ENV 对齐。
COUNT_ENV = {
    "embedding": "A1_EMBEDDING_COUNT_FILE",
    "sdpa_math": "A1_SDPA_MATH_COUNT_FILE",
}


class _FileCounter(dict):
    """register_a1(counter=...) 的 dict 计数器: 每次更新写计数文件。"""

    def __init__(self, path: str, op: str):
        super().__init__(n=0)
        self._path = path
        self._op = op

    def __setitem__(self, key, value):
        import json
        super().__setitem__(key, value)
        try:
            with open(self._path, "w") as f:
                json.dump({self._op: int(self.get("n", 0))}, f)
        except OSError:
            pass


def _ops() -> list[str]:
    raw = os.environ.get("FLAGOS_A1_OPS")
    if raw is None:
        return list(_DEFAULT_OPS)
    return [x for x in raw.split(",") if x]


def _purge_kernel() -> None:
    # kernel: 各 op 同名 kernel 包；reference: sdpa_math/kernel/torch_level
    # 等用裸名 `from reference import ...` 导入各 op 自己的 reference.py，
    # 上一个算子留下的缓存会让后加载算子拿到错模块（实测 sdpa_math 因此
    # LOAD FAIL）→ 一并清理。
    for m in [m for m in list(sys.modules)
              if m == "kernel" or m.startswith("kernel.")
              or m == "reference" or m.startswith("reference.")]:
        del sys.modules[m]


def _load_isolated(root: Path, op: str):
    opd = root / "ops" / op
    reg = opd / "register.py"
    if not reg.is_file():
        return None
    _purge_kernel()
    sys.path.insert(0, str(opd))
    try:
        spec = importlib.util.spec_from_file_location(f"_a1reg_{op}", reg)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for sub in ("kernel", "kernel.torch_level", "kernel.triton_level"):
            try:
                importlib.import_module(sub)
            except Exception:
                pass
        saved = {n: sys.modules[n] for n in list(sys.modules)
                 if n == "kernel" or n.startswith("kernel.")}
    except Exception:
        return None
    finally:
        try:
            sys.path.remove(str(opd))
        except ValueError:
            pass
    return mod, saved


def _wrap(fn, saved):
    def wrapper(*args, **kwargs):
        old = {}
        for n in saved:
            old[n] = sys.modules.get(n)
            sys.modules[n] = saved[n]
        try:
            return fn(*args, **kwargs)
        finally:
            for n, v in old.items():
                if v is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = v

    wrapper.__name__ = getattr(fn, "__name__", "wrapper")
    return wrapper


def register_all(dispatch_key: str = "PrivateUse1",
                 root: str | None = None) -> list[str]:
    """注册本仓库的 A1 aten 算子，返回成功注册的算子名列表。"""
    base = Path(root or os.environ.get("FLAGOS_TEMPLATES_ROOT") or _DEFAULT_ROOT)
    registered: list[str] = []
    for op in _ops():
        r = _load_isolated(base, op)
        if r is None:
            continue
        mod, saved = r
        for name in list(vars(mod)):
            fn = getattr(mod, name, None)
            if callable(fn) and name.endswith("_aten"):
                setattr(mod, name, _wrap(fn, saved))
        try:
            counter = None
            var = COUNT_ENV.get(op)
            file_base = os.environ.get(var) if var else None
            if file_base:
                counter = _FileCounter(f"{file_base}.{os.getpid()}", op)
            kwargs = {}
            impl = os.environ.get(f"A1_{op.upper()}_IMPL")
            if impl:
                kwargs["impl"] = impl
            mod.register_a1(dispatch_key, counter=counter, **kwargs)
            registered.append(op)
        except Exception:
            pass
    if "sdpa_math" in registered:
        _install_sdpa_consumer()
    return registered


def _tolist(x) -> list[int]:
    return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]


def _append_evidence(record: dict) -> None:
    path = os.environ.get("A1_SDPA_EVIDENCE_FILE")
    if not path:
        return
    import json
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _prob_stats(probs) -> dict:
    """概率图统计（双输出契约的 P）: 行和偏差 / 对角线 / 熵。"""
    import torch  # noqa: F811
    p = probs.detach().float()
    if p.dim() == 3:
        p = p.unsqueeze(0)
    rowsum_dev = (p.sum(-1) - 1.0).abs().max().item()
    diag = torch.diagonal(p, dim1=-2, dim2=-1)
    pc = p.clamp_min(1e-12)
    entropy = (-(pc * pc.log()).sum(-1)).mean().item()
    return {
        "probs_shape": list(p.shape),
        "probs_rowsum_max_dev": rowsum_dev,
        "probs_diag_mean": diag.mean().item(),
        "probs_entropy_mean": entropy,
    }


def _install_sdpa_consumer() -> bool:
    """把真实 vLLM 的 prefill 注意力接到 A1 private math（等价消费者）。

    Kunlunxin vendor 的 prefill_attention 不经过 F.sdpa，因此真实文本模型
    里没有任何 aten::scaled_dot_product_attention 消费点。这里在
    EngineCore 进程内把 KunlunxinAttentionBackendImpl.context_forward 的
    「无前缀缓存」分支改为逐请求调用 A1 的
    torch.ops.aten._scaled_dot_product_attention_math（双输出 O+P 直接喂
    回模型输出缓冲），decode 与前缀缓存分支保持 vendor 原路径。
    每次消费把概率图统计追加到 A1_SDPA_EVIDENCE_FILE（JSONL）。
    """
    if os.environ.get("A1_SDPA_CONSUMER") != "prefill":
        return False
    import torch  # noqa: F811
    try:
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
            KunlunxinAttentionBackendImpl,
        )
    except Exception:
        return False
    if getattr(KunlunxinAttentionBackendImpl, "_a1_consumer_installed", False):
        return True
    orig = KunlunxinAttentionBackendImpl.context_forward

    def context_forward(self, query, key, value, out, attn_metadata,
                        *args, **kwargs):
        reason = None
        try:
            is_causal = bool(kwargs.get("is_causal", True))
            if args:
                reason = "positional_extra_args"
            elif query is None or key is None or value is None or out is None:
                reason = "missing_tensors"
            elif attn_metadata is None:
                reason = "no_metadata"
            elif getattr(attn_metadata, "slot_mapping", None) is None:
                # vLLM 预热/profiling: forward() 构造的 fake-meta
                # （slot_mapping=None, 16k dummy tokens）→ 保持 vendor 原路径，
                # 不让 16k² 概率图污染证据统计。
                reason = "profiling_fake_meta"
            elif (self.sliding_window is not None
                  and self.sliding_window > 0):
                reason = "sliding_window"
            else:
                qsl_h = getattr(attn_metadata, "query_start_loc_host", None)
                kvpl_h = getattr(attn_metadata, "kv_prefix_start_loc_host",
                                 None)
                if qsl_h is None or kvpl_h is None:
                    reason = "no_lod"
                elif _tolist(qsl_h)[-1] != _tolist(kvpl_h)[-1]:
                    reason = "prefix_cache"
        except Exception as e:
            reason = f"check_error:{type(e).__name__}"

        if reason is not None:
            _append_evidence({"type": "fallback", "reason": reason})
            return orig(self, query, key, value, out, attn_metadata,
                        *args, **kwargs)

        qsl = _tolist(attn_metadata.query_start_loc)
        enable_gqa = self.num_heads != self.num_kv_heads
        n_calls = 0
        for i in range(len(qsl) - 1):
            s, e = qsl[i], qsl[i + 1]
            if e <= s:
                continue
            Qi = query[s:e].transpose(0, 1).unsqueeze(0)
            Ki = key[s:e].transpose(0, 1).unsqueeze(0)
            Vi = value[s:e].transpose(0, 1).unsqueeze(0)
            oi, probs = torch.ops.aten._scaled_dot_product_attention_math(
                Qi, Ki, Vi, None, 0.0, is_causal, None,
                scale=self.scale, enable_gqa=enable_gqa)
            out[s:e].copy_(oi.squeeze(0).transpose(0, 1).to(out.dtype))
            n_calls += 1
            try:
                rec = {"type": "consume", "req": i, "qlen": e - s,
                       "heads": self.num_heads,
                       "kv_heads": self.num_kv_heads,
                       "head_size": self.head_size,
                       "is_causal": is_causal}
                rec.update(_prob_stats(probs))
                _append_evidence(rec)
            except Exception:
                pass
        if n_calls:
            return out.view(-1, self.num_heads * self.head_size)
        return orig(self, query, key, value, out, attn_metadata,
                    *args, **kwargs)

    KunlunxinAttentionBackendImpl.context_forward = context_forward
    KunlunxinAttentionBackendImpl._a1_consumer_installed = True
    return True


def grad_probe() -> dict:
    """A1 private math 的框架级梯度证据（在真实引擎同一注册进程内跑）。

    bf16 XPU 前向+双输出反向（dout+dprobs 各为全 1，覆盖 dP 消费者路径），
    对照 CPU fp64 原生 math 的前向/反向，记录 max_abs/max_rel。
    """
    if os.environ.get("A1_SDPA_GRADPROBE") != "1":
        return {}
    import torch  # noqa: F811
    old = os.environ.get("SDPA_MATH_P800_FUSED_BWD")
    os.environ["SDPA_MATH_P800_FUSED_BWD"] = "0"  # 走纯数学反向，XPU 安全
    try:
        torch.manual_seed(1234)
        B, Hq, Hkv, S, D = 1, 8, 4, 24, 64
        dev = "cuda"
        dt = torch.bfloat16
        q = torch.randn(B, Hq, S, D, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn(B, Hkv, S, D, device=dev, dtype=dt, requires_grad=True)
        v = torch.randn(B, Hkv, S, D, device=dev, dtype=dt, requires_grad=True)
        out, probs = torch.ops.aten._scaled_dot_product_attention_math(
            q, k, v, None, 0.0, True, None, enable_gqa=True)
        (out.float().sum() + probs.float().sum()).backward()

        qc = q.detach().cpu().double().requires_grad_()
        kc = k.detach().cpu().double().requires_grad_()
        vc = v.detach().cpu().double().requires_grad_()
        outr, probsr = torch.ops.aten._scaled_dot_product_attention_math(
            qc, kc, vc, None, 0.0, True, None, enable_gqa=True)
        (outr.sum() + probsr.sum()).backward()

        def cmp(a, b):
            a = a.detach().float().cpu()
            b = b.detach().float().cpu()
            d = (a - b).abs()
            denom = b.abs().clamp_min(1e-6)
            return round(d.max().item(), 6), round((d / denom).max().item(), 6)

        res = {"ok": True, "dtype": "bf16", "ref": "cpu_float64_native_math"}
        res["out_max_abs"], res["out_max_rel"] = cmp(out, outr)
        res["probs_max_abs"], res["probs_max_rel"] = cmp(probs, probsr)
        res["dq_max_abs"], res["dq_max_rel"] = cmp(q.grad, qc.grad)
        res["dk_max_abs"], res["dk_max_rel"] = cmp(k.grad, kc.grad)
        res["dv_max_abs"], res["dv_max_rel"] = cmp(v.grad, vc.grad)
        return res
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    finally:
        if old is None:
            os.environ.pop("SDPA_MATH_P800_FUSED_BWD", None)
        else:
            os.environ["SDPA_MATH_P800_FUSED_BWD"] = old
        torch.cuda.empty_cache()
