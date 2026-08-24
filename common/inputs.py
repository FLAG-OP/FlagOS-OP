# 测试输入生成 scheme 库（框架级文本 + 算子级 token 双形态）。
#
# 设计目标: 输入可声明、可复现、可跨硬件——同一 spec 在 CPU/NVIDIA/
# P800 上生成完全相同的输入，黄金输出才具备可比性。
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


# ---------- 框架级: 文本 prompt ----------
def expand_template(template: str, variables: list[str]) -> list[str]:
    return [template.format(item=item) for item in variables]


# ---------- 算子级: token id 输入 ----------
def deterministic_tokens(request_index: int, input_len: int) -> list[int]:
    """确定性不重复 token 序列（与历史基准脚本同 scheme，保证可比）。

    (1000 + i*97 + p*13) % 100000
    """
    return [
        (1000 + request_index * 97 + position * 13) % 100000
        for position in range(input_len)
    ]


def seeded_random_tokens(seed: int, request_index: int, input_len: int,
                         vocab: int = 100000) -> list[int]:
    rng = random.Random(f"{seed}:{request_index}")
    return [rng.randrange(vocab) for _ in range(input_len)]


def tokenize_clamped(tokenizer, prompt: str, vocab_size: int) -> list[int]:
    """tokenize 并钳制到模型词表内。

    背景: 随机权重模型的 tokenizer 词表(151669)大于模型 embedding
    (128256)。vLLM/XPU 路径的 embedding 查找不做边界检查（静默越界
    读），CPU transformers 会正确报 IndexError。跨设备可比的前提是
    双端输入完全一致且合法，故统一 tokenize 后按词表取模钳制。
    """
    return [i % vocab_size for i in tokenizer(prompt)["input_ids"]]


def generate_from_spec(spec: dict) -> dict:
    """按 spec 生成全部输入。返回 {"framework": [...], "operator": {...}}"""
    out: dict = {"framework": [], "operator": {}}

    for cat in spec.get("framework", []):
        if not cat.get("enabled", True):
            continue
        kind = cat["kind"]
        if kind == "fixed":
            out["framework"].extend(cat["prompts"])
        elif kind == "length_sweep":
            out["framework"].extend(
                expand_template(cat["template"], cat["vars"]))
        elif kind == "edge_cases":
            out["framework"].extend(cat["prompts"])
        else:
            raise ValueError(f"未知 framework 输入类别: {kind}")

    seed = spec.get("seed", 0)
    for cat in spec.get("operator", []):
        if not cat.get("enabled", True):
            continue
        kind = cat["kind"]
        if kind == "token_based":
            scheme = cat.get("scheme", "deterministic")
            gen = (deterministic_tokens if scheme == "deterministic"
                   else lambda i, n: seeded_random_tokens(seed, i, n))
            out["operator"][cat.get("name", "default")] = {
                "scheme": scheme,
                "input_len": cat["input_len"],
                "prompts": [
                    gen(i, cat["input_len"])
                    for i in range(cat.get("count", 1))
                ],
            }
        else:
            raise ValueError(f"未知 operator 输入类别: {kind}")
    return out
