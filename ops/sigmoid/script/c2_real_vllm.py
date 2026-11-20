#!/usr/bin/env python3
"""采集 sigmoid 在 P800 FlagGems 5.0 真实 vLLM EngineCore 的 C2 证据。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
FRAMEWORK = ROOT / "tests" / "framework_level"
for path in (ROOT, FRAMEWORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# wt-2026-11-20-fix
# wt <liyanze1560495@hotmail.com>


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(path: Path, bundle: dict, evidence_path: Path) -> None:
    acceptance = bundle["acceptance"]
    environment = bundle["environment"]
    result = bundle["result"]
    evidence_link = Path(os.path.relpath(
        evidence_path.resolve(), path.parent.resolve())).as_posix()
    processes = ", ".join(
        f"PID {item['pid']}={item['hits']} hits"
        for item in result.get("count_processes", [])
        if item.get("op") == "sigmoid" and item.get("hits", 0) > 0
    ) or "无"
    tokens_base = json.dumps(
        result.get("baseline_output_token_ids"), ensure_ascii=False)
    tokens_injected = json.dumps(
        result.get("injected_output_token_ids"), ensure_ascii=False)
    text = f"""# aten.sigmoid P800 真实 vLLM C2 报告

> wt-2026-11-20-fix · wt <liyanze1560495@hotmail.com>

| 项 | 结果 |
|---|---|
| profile | `{bundle['profile']}` |
| 状态 | **{bundle['status']}** |
| consumer | `{bundle['consumer']}`（真实 vLLM EngineCore） |
| 模型 | `{result.get('model')}` |
| 提交 | `{bundle['commit']}` |
| 采集时间 | `{bundle['collected_at']}` |
| 默认 / 验证路由 | `{bundle['default_route']}` / `{bundle['verification_route']}` |
| 物理可见卡 | `{environment['visible_devices']}`；TP={environment['tensor_parallel_size']} |
| sigmoid 命中 | `{acceptance['sigmoid_hits']}`；{processes} |
| worker 命中 | `{acceptance['worker_process_hit']}`（命中 PID 与 runner PID 分离） |
| 输出一致 | `{acceptance['output_match']}` |

## 消费路径

基线组和注入组都在 `vllm.v1.sample.Sampler.forward` 中对逐行归一化的
logits 消费 `aten::sigmoid`；greedy sampling 下 sigmoid 单调变换保持
argmax 排序。基线使用原生 sigmoid，注入组显式选择 Triton sigmoid。
命中计数来自 EngineCore worker 子进程，不是外层编排进程。

## 输出对拍

- baseline token ids: `{tokens_base}`
- injected token ids: `{tokens_injected}`
- consumer records: `{acceptance['consumer_records']}`

## 环境与复现

- Python `{environment['python']}`
- PyTorch `{environment['torch']}`
- Triton `{environment['triton']}`
- FlagGems `{environment['flag_gems']}`
- vLLM `{environment['vllm']}`
- vLLM-FL `{environment['vllm_fl']}`
- 环境锁 `{environment['env_lock']}`（SHA256 `{environment['env_lock_sha256']}`）
- 设备 profile `{environment['device_profile']}`（SHA256 `{environment['device_profile_sha256']}`）

```bash
{bundle['command']}
```

结构化原始证据：[JSON]({evidence_link})。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="p800-kunlunxin-gems5")
    parser.add_argument(
        "--model",
        help="真实 vLLM 模型目录；省略时使用 FLAGOS_MODEL_PATH 或 tiny 模型",
    )
    parser.add_argument(
        "--out",
        default=str(OP_DIR / "evidence" / "c2_p800-kunlunxin-gems5.json"),
    )
    parser.add_argument(
        "--report",
        default=str(OP_DIR / "reports" / "c2_p800-kunlunxin-gems5.md"),
    )
    args = parser.parse_args()

    if args.model:
        os.environ["FLAGOS_MODEL_PATH"] = args.model

    from common.device import load_profile
    from _aten_harness import run_real_vllm

    profile = load_profile(args.profile)
    os.environ["FLAGOS_A1_OPS"] = "sigmoid"
    os.environ["FLAGOS_A1_IMPL"] = "triton"
    result = run_real_vllm(profile)

    sigmoid_hits = int(result.get("counts", {}).get("sigmoid", 0))
    injected_runner_pid = result.get("runner_pids", {}).get("injected")
    worker_records = [
        item for item in result.get("count_processes", [])
        if item.get("op") == "sigmoid" and item.get("hits", 0) > 0
    ]
    worker_only = bool(worker_records) and all(
        item.get("pid") != injected_runner_pid for item in worker_records
    )
    consumer_records = result.get("sigmoid_consumer", [])
    passed = bool(
        result.get("output_match")
        and sigmoid_hits > 0
        and worker_only
        and consumer_records
    )

    lock = ROOT / "configs" / "env" / f"{profile.name}.lock.yaml"
    device_profile = ROOT / "configs" / "devices" / f"{profile.name}.yaml"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    output = Path(args.out)
    try:
        output_label = output.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        output_label = str(output)
    model_arg = f" --model {args.model}" if args.model else ""
    command = (
        "python3 ops/sigmoid/script/c2_real_vllm.py "
        f"--profile {profile.name}{model_arg} --out {output_label}"
    )
    bundle = {
        "schema": 1,
        "operator": "aten::sigmoid",
        "status": "PASS" if passed else "FAIL",
        "ok": passed,
        "evidence_level": "C2",
        "consumer_type": "real-vllm-enginecore",
        "consumer": "vllm.v1.sample.Sampler.forward logits",
        "consumer_contract": (
            "baseline=native sigmoid; injected=Triton sigmoid; both consume "
            "normalized logits before greedy sampling"
        ),
        "profile": profile.name,
        "commit": commit,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "default_route": "native",
        "verification_route": "triton",
        "environment": {
            "python": sys.version.split()[0],
            "torch": _version("torch"),
            "triton": _version("triton"),
            "flag_gems": _version("flag-gems"),
            "vllm": _version("vllm"),
            "vllm_fl": _version("vllm-plugin-fl"),
            "env_lock": str(lock.relative_to(ROOT)),
            "env_lock_sha256": _sha256(lock),
            "device_profile": str(device_profile.relative_to(ROOT)),
            "device_profile_sha256": _sha256(device_profile),
            "torch_device": profile.torch_device,
            "visible_devices": profile.visible_devices,
            "tensor_parallel_size": profile.framework.get(
                "tensor_parallel_size"),
            "distributed_executor_backend": profile.framework.get(
                "distributed_executor_backend"),
        },
        "result": result,
        "acceptance": {
            "output_match": bool(result.get("output_match")),
            "sigmoid_hits": sigmoid_hits,
            "worker_process_hit": worker_only,
            "consumer_records": len(consumer_records),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    _write_report(Path(args.report), bundle, output)
    print(json.dumps(bundle["acceptance"], ensure_ascii=False))
    print(f"C2 {bundle['status']}: {output}")
    print(f"report: {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
