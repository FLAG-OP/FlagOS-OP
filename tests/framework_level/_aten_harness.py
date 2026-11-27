# 真实 vLLM 的 A1 aten 多算子框架级验证编排。
#
# 做什么:
#   1. 生成/复用一个小模型（FLAGOS_MODEL_PATH 优先，否则临时生成 tiny Llama）。
#   2. 基线跑一次真实 vLLM 前向（无注入）。
#   3. 注入跑一次（sitecustomize 在每个 vLLM 子进程内注册本仓库的 9 个
#      A1 aten 算子）。
#   4. 断言: 输出逐位一致；且至少一个被注入算子被真实前向消费（计数 > 0）。
#
# 这是「第 4 步 应用层」的落地: 先前的 ops/<op>/test/framework_level.py
# 用等价 torch API 子进程模拟，这里换成真实 vLLM。
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
INJECTION = ROOT / "injection"

# wt-2026-11-20-fix
# wt <liyanze1560495@hotmail.com>

# 算子 → 计数文件环境变量（与 ops/<op>/register.py 一致）
COUNT_ENV = {
    "type_as": "A1_TYPE_AS_COUNT_FILE",
    "clone": "A1_CLONE_COUNT_FILE",
    "contiguous": "A1_CONTIGUOUS_COUNT_FILE",
    "copy_": "A1_COPY_COUNT_FILE",
    "dropout": "A1_DROPOUT_COUNT_FILE",
    "empty_like": "A1_EMPTY_LIKE_COUNT_FILE",
    "empty_strided": "A1_EMPTY_STRIDED_COUNT_FILE",
    "item": "A1_ITEM_COUNT_FILE",
    "_local_scalar_dense": "A1_LSD_COUNT_FILE",
    "embedding": "A1_EMBEDDING_COUNT_FILE",
    "sdpa_math": "A1_SDPA_MATH_COUNT_FILE",
    "sigmoid": "A1_SIGMOID_COUNT_FILE",
}

# 计数分片文件名使用仓库算子名，JSON payload 则有两种历史格式：
# 多数算子写短名，embedding/sdpa_math 写完整 aten schema。显式列出
# canonical key 和兼容 key，避免把 PID、时间戳等整数元数据误算成命中。
COUNT_PAYLOAD_KEYS = {
    op: (op,) for op in COUNT_ENV
}
COUNT_PAYLOAD_KEYS.update({
    "embedding": ("aten::embedding", "embedding"),
    "sdpa_math": (
        "aten::_scaled_dot_product_attention_math", "sdpa_math"),
    "sigmoid": ("sigmoid", "aten::sigmoid"),
})


def vllm_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("vllm") is not None


def ensure_model() -> str:
    """返回可用模型路径。FLAGOS_MODEL_PATH 优先，否则生成 tiny Llama。"""
    env = os.environ.get("FLAGOS_MODEL_PATH")
    if env and Path(env, "config.json").is_file():
        return env
    d = Path(tempfile.gettempdir()) / "flagos_tiny_llama"
    if (d / "config.json").is_file():
        return str(d)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)

    words = ["<unk>", "<s>", "</s>", "<pad>", "the", "future", "of", "AI",
             "is", "bright", "hello", "world", "a", "b", "c", "1", "2", "3",
             "machine", "learning", "model", "data", "test", "prompt"]
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>",
                            eos_token="</s>", unk_token="<unk>",
                            pad_token="<pad>").save_pretrained(str(d))
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64,
                      intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=256, bos_token_id=1,
                      eos_token_id=2, pad_token_id=3)
    import torch
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(str(d))
    return str(d)


def _run(model: str, out_json: str, *, inject: bool, count_dir: str,
         dispatch_key: str, extra_env: dict | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{INJECTION}{os.pathsep}{ROOT}"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    for k, v in (extra_env or {}).items():
        env[k] = v
    if inject:
        env["FLAGOS_A1_ATEN_INJECT"] = "1"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["A1_DISPATCH_KEY"] = dispatch_key
        for op, var in COUNT_ENV.items():
            env[var] = str(Path(count_dir) / op)
    cmd = [sys.executable, str(HERE / "_aten_runner.py"), str(out_json), model]
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return json.loads(Path(out_json).read_text())


def _count_value(op: str, payload) -> int | None:
    """返回单个算子分片的合法命中数；格式不合法时返回 ``None``。"""
    if not isinstance(payload, dict):
        return None
    for key in COUNT_PAYLOAD_KEYS.get(op, (op,)):
        if key not in payload:
            continue
        value = payload[key]
        if (isinstance(value, int) and not isinstance(value, bool)
                and value >= 0):
            return value
        return None
    return None


def _agg_counts(count_dir: str) -> dict:
    agg: dict[str, int] = {}
    for p in glob.glob(str(Path(count_dir) / "*")):
        op = os.path.basename(p)
        stem, dot, tail = op.rpartition(".")
        if dot and tail.isdigit():      # 计数文件带 pid 后缀 → 还原算子名
            op = stem
        try:
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
            value = _count_value(op, payload)
            if value is not None:
                agg[op] = agg.get(op, 0) + value
        except (OSError, ValueError, TypeError):
            continue
    return agg


def _count_processes(count_dir: str) -> list[dict]:
    details = []
    for p in glob.glob(str(Path(count_dir) / "*")):
        name = os.path.basename(p)
        stem, dot, tail = name.rpartition(".")
        if not (dot and tail.isdigit()):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError, TypeError):
            continue
        value = _count_value(stem, payload)
        if value is None:
            continue
        details.append({"op": stem, "pid": int(tail),
                        "hits": value, "file": name})
    return sorted(details, key=lambda item: (item["op"], item["pid"]))


def _jsonl_records(path: str) -> list[dict]:
    records = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return records


def run_real_vllm(profile) -> dict:
    model = ensure_model()
    with tempfile.TemporaryDirectory() as d:
        out_base = os.path.join(d, "base.json")
        out_inj = os.path.join(d, "inj.json")
        count_dir = os.path.join(d, "counts")
        os.makedirs(count_dir, exist_ok=True)
        extra_env = {}
        selected_devices = os.environ.get(
            "CUDA_VISIBLE_DEVICES", profile.visible_devices)
        common_env = {
            # 与正式 profile 的物理卡选择和多进程执行方式保持一致；
            # runner 仍用 128 token / 0.2 显存比例的最小 C2 负载。
            "CUDA_VISIBLE_DEVICES": selected_devices,
            "XPU_VISIBLE_DEVICES": os.environ.get(
                "XPU_VISIBLE_DEVICES", selected_devices),
            "A1_VLLM_TP": str(profile.framework.get(
                "tensor_parallel_size", 1)),
            "A1_VLLM_EXECUTOR": str(profile.framework.get(
                "distributed_executor_backend", "mp")),
        }
        ops = [o.strip() for o in
               os.environ.get("FLAGOS_A1_OPS", "").split(",") if o.strip()]
        if "sigmoid" in ops:
            common_env["A1_SIGMOID_CONSUMER"] = "sampler_logits"
            common_env["A1_SIGMOID_EVIDENCE_FILE"] = os.path.join(
                count_dir, "sigmoid_consumer.jsonl")
        if profile.name.startswith("p800"):
            # P800 上 sdpa_math 走 vendor O/LSE + exact-P 实现（与 op 层交付一致）
            extra_env["A1_SDPA_MATH_IMPL"] = "p800"
            if "sdpa_math" in ops:
                # 真实引擎等价消费者: prefill 注意力 → A1 private math；
                # 证据 = 命中计数 + 概率图统计 JSONL + 梯度 probe。
                extra_env["A1_SDPA_CONSUMER"] = "prefill"
                extra_env["A1_SDPA_EVIDENCE_FILE"] = os.path.join(
                    count_dir, "sdpa_evidence.jsonl")
                extra_env["A1_SDPA_GRADPROBE"] = "1"
        base = _run(model, out_base, inject=False, count_dir=count_dir,
                    dispatch_key=profile.dispatch_key,
                    extra_env=common_env)
        inject_env = dict(common_env)
        inject_env.update(extra_env)
        inj = _run(model, out_inj, inject=True, count_dir=count_dir,
                    dispatch_key=profile.dispatch_key, extra_env=inject_env)
        counts = _agg_counts(count_dir)
        count_processes = _count_processes(count_dir)
        sigmoid_consumer = _jsonl_records(
            os.path.join(count_dir, "sigmoid_consumer.jsonl"))
        sdpa = _agg_sdpa_evidence(count_dir, inj)

    match = base["output_token_ids"] == inj["output_token_ids"]
    total = sum(counts.values())
    return {
        "ok": bool(match and total > 0),
        "engine": "vllm",
        "profile": profile.name,
        "model": model,
        "output_match": match,
        "baseline_output_token_ids": base["output_token_ids"],
        "injected_output_token_ids": inj["output_token_ids"],
        "counts": counts,
        "count_processes": count_processes,
        "total_calls": total,
        "runner_pids": {"baseline": base.get("runner_pid"),
                        "injected": inj.get("runner_pid")},
        "sigmoid_consumer": sigmoid_consumer,
        "sdpa": sdpa,
        "latency_s": {"base": base.get("latency_s"),
                      "inj": inj.get("latency_s")},
    }


def _agg_sdpa_evidence(count_dir: str, inj: dict) -> dict:
    """汇 sdpa_math 消费/概率图证据 + 注入运行的梯度 probe 结果。"""
    path = Path(count_dir) / "sdpa_evidence.jsonl"
    sdpa: dict = {}
    if path.is_file():
        import json as _json
        recs = []
        for line in path.read_text().splitlines():
            try:
                recs.append(_json.loads(line))
            except ValueError:
                continue
        consumed = [r for r in recs if r.get("type") == "consume"]
        fallbacks = [r for r in recs if r.get("type") == "fallback"]
        sdpa["consumer"] = "vllm_prefill_attention"
        sdpa["calls"] = len(consumed)
        sdpa["fallbacks"] = len(fallbacks)
        sdpa["fallback_reasons"] = sorted(
            {r.get("reason", "?") for r in fallbacks})
        if consumed:
            sdpa["prob_sample"] = consumed[0]
    if isinstance(inj.get("sdpa_grad"), dict):
        sdpa["grad"] = inj["sdpa_grad"]
    return sdpa
