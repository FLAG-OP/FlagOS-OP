# 框架级测试编排工具: 基线/插件双跑 + 结果比对
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
GOLDEN_DIR = ROOT / "golden"


def _count_file(route: str) -> str:
    return {
        "a1": "/tmp/a1_aten_counts.json",
        "a2": "/tmp/a2_silu_counts.json",
        "b": "/tmp/audit_vendor_counts.json",
    }[route]


def _plugin_env(route: str, profile) -> dict:
    env: dict[str, str] = {}
    if route == "a2":
        env["VLLM_FL_PLUGIN_MODULES"] = "routes.a2_dispatch.plugin.register_ops"
        env["A2_SILU_COUNT_FILE"] = _count_file("a2")
    elif route == "b":
        env["VLLM_FL_PLUGIN_MODULES"] = "routes.b_vendor.backend.register_ops"
        env["AUDIT_VENDOR_COUNT_FILE"] = _count_file("b")
        # ⚠️ 框架级恒等断言必须用 reference 委托: 实测本机
        # xtorch_ops.swiglu 独立 eager 调用不写 out（哨兵验证），
        # vendor 委托无法保证数值恒等。
        env["AUDIT_DELEGATE"] = "reference"
        pkg, func = profile.delegate_import()
        if pkg and func:
            env["AUDIT_VENDOR_DELEGATE"] = f"{pkg}.{func}"
    elif route == "a1":
        env["A1_ATEN_COUNT_FILE"] = _count_file("a1")
        # sitecustomize 注入: 每个 vLLM 子进程启动时注册 aten::silu;
        # FlagGems 也会注册 silu，需加入 torch 层黑名单防止覆盖计数实现
        env["FLAGOS_TEMPLATES_A1_INJECT"] = "1"
        env["FLAGOS_TEMPLATES_ROOT"] = str(ROOT)
        env["A1_DISPATCH_KEY"] = profile.dispatch_key
        env["VLLM_FL_FLAGOS_BLACKLIST"] = "silu,silu_"
    return env


def _baseline_env(route: str, profile) -> dict:
    """基线跑的环境。a1 需同样钉住 reference 路径，使基线与插件跑
    之间唯一差异是 aten::silu 覆盖本身（否则会混入 vendor↔reference
    的数值差，exact 断言不成立）。"""
    if route in ("a1", "b"):
        return {"VLLM_FL_PER_OP": "silu_and_mul=reference"}
    return {}


def run_case(route: str, profile, out_name: str, with_plugin: bool) -> dict:
    """跑一次 vLLM 子进程并返回结果 dict。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'injection'}{os.pathsep}{ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = profile.visible_devices
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    # 统一 token 钳制输入: 与黄金生成一致，且规避 vendor 栈
    # embedding 静默越界（tokenizer 151669 > embedding 128256）
    env["PROMPTS_AS_TOKENS"] = "1"
    # keep_default_prefer=true: 保留继承的 VLLM_FL_PREFER（P800 保护路径，
    # 容器默认非法值恰好让算子回落 vendor kernel）
    # keep_default_prefer=false: 清掉继承值，走 config 默认
    if profile.framework_quirks.get("keep_default_prefer"):
        pass  # 保留继承值
    else:
        env.pop("VLLM_FL_PREFER", None)
    env.pop("VLLM_FL_PER_OP", None)
    if not with_plugin:
        env.update(_baseline_env(route, profile))
    if with_plugin:
        env.update(_plugin_env(route, profile))

    out_path = HERE / out_name
    cmd = [sys.executable, str(HERE / "_vllm_runner.py"), str(out_path),
           "--device", profile.name, "--route", route]
    if with_plugin:
        cmd.append("--with-plugin")
    print(f"[fw:{route}] running {'WITH' if with_plugin else 'WITHOUT'} plugin ...")
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return json.loads(out_path.read_text())


def reset_counts(route: str) -> None:
    """清空该 route 的全部 pid 分片计数文件。"""
    import glob as _glob
    for p in _glob.glob(f"{_count_file(route)}.*"):
        Path(p).unlink(missing_ok=True)
    Path(_count_file(route)).unlink(missing_ok=True)


def read_counts(route: str) -> dict:
    """汇总 pid 分片 + 兼容单文件旧格式。"""
    import glob as _glob
    total: dict = {}
    for p in sorted(_glob.glob(f"{_count_file(route)}.*")) + [_count_file(route)]:
        try:
            with open(p) as f:
                for k, v in json.load(f).items():
                    total[k] = total.get(k, 0) + v
        except (OSError, ValueError):
            continue
    return total


def compare_outputs(base: dict, plug: dict, *, exact: bool,
                    k_tokens: int = 2, min_match_prompts: int = 2,
                    exact_prefix: int = 8) -> dict:
    """输出比对。

    exact=True          : 要求前 exact_prefix 个 token 完全一致
                          （数值恒等委托适用; 本栈跨进程后期 token
                          存在偶发非确定性，不能要求全量一致）
    exact=False         : 前 k token 一致率（自定义数值实现适用，
                          随机权重贪心解码允许混沌分叉）
    返回 {mode, matched, detail}
    """
    b, p = base["output_token_ids"], plug["output_token_ids"]
    if exact:
        prefixes = exact_prefixes_per_prompt(base.get("device", ""),
                                             cap=exact_prefix)
        ok = all(x[:k] == y[:k]
                 for x, y, k in zip(b, p, prefixes * len(b)))
        return {"mode": f"exact-prefix-per-prompt{prefixes}", "ok": ok,
                "detail": ("per-prompt prefixes identical" if ok
                           else f"per-prompt prefixes {prefixes} differ")}
    matched = sum(1 for x, y in zip(b, p) if x[:k_tokens] == y[:k_tokens])
    return {"mode": f"first-{k_tokens}-token", "ok": matched >= min_match_prompts,
            "detail": f"{matched}/{len(b)} prompts matched first {k_tokens} tokens"}



def exact_prefixes_per_prompt(device_name: str, cap: int = 8) -> list[int]:
    """逐 prompt 自适应前缀: max(1, min(cap, 黄金稳定长度 - 1))。

    黄金稳定长度是"已观测到分叉的位置"，断言取 length-1 避免踩线。
    """
    golden = load_golden(device_name)
    if golden is None:
        return [cap]
    return [max(1, min(cap, s - 1)) for s in golden["stable_prefix_lens"]]


def load_golden(device_name: str) -> Optional[dict]:
    p = GOLDEN_DIR / f"{device_name}_golden.json"
    return json.loads(p.read_text()) if p.exists() else None


def check_golden(result: dict, prefix: int = 8) -> dict:
    """将本次输出与黄金共识前缀比对（设备无 golden 文件时跳过）。

    用途: 跨会话回归检测——即使本次测试的基线/插件一致，
    若相对历史黄金输出发生漂移（FlagOS/torch/厂商栈升级）也会报警。
    """
    golden = load_golden(result.get("device", ""))
    if golden is None:
        return {"mode": "golden", "ok": True, "detail": "no golden file, skipped"}

    outs = result["output_token_ids"]
    consensus = golden["consensus_prefix"]
    k = min(prefix, min(golden["stable_prefix_lens"]))
    ok = len(outs) == len(consensus) and all(
        o[:k] == c[:k] for o, c in zip(outs, consensus))
    return {"mode": f"golden-prefix-{k}", "ok": ok,
            "detail": (f"matches golden consensus (prefix {k})"
                       if ok else f"DRIFT vs golden (prefix {k})")}
