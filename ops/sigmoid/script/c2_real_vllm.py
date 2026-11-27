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


def _positive_int(value) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value > 0)


def _validate_run(name: str, run) -> list[str]:
    """校验一次 vLLM 生成确实完成预期请求并产出有效 token。"""
    errors: list[str] = []
    if not isinstance(run, dict):
        return [f"{name}: run record missing"]
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        errors.append(f"{name}: run_id missing")
    expected = run.get("expected_request_count")
    generated = run.get("generated_request_count")
    outputs = run.get("output_token_ids")
    if not _positive_int(expected):
        errors.append(f"{name}: expected_request_count must be positive")
    if not isinstance(generated, int) or isinstance(generated, bool):
        errors.append(f"{name}: generated_request_count invalid")
    elif _positive_int(expected) and generated != expected:
        errors.append(
            f"{name}: generated {generated}/{expected} requests")
    if not isinstance(outputs, list):
        errors.append(f"{name}: output_token_ids missing")
        outputs = []
    elif _positive_int(expected) and len(outputs) != expected:
        errors.append(
            f"{name}: output count {len(outputs)} != {expected}")
    valid_total = 0
    for index, token_ids in enumerate(outputs):
        if (not isinstance(token_ids, list) or not token_ids
                or any(not isinstance(token, int) or isinstance(token, bool)
                       for token in token_ids)):
            errors.append(f"{name}: request {index} has invalid/empty tokens")
            continue
        valid_total += len(token_ids)
    recorded_total = run.get("generated_token_count")
    if not _positive_int(recorded_total) or recorded_total != valid_total:
        errors.append(
            f"{name}: generated_token_count {recorded_total!r} "
            f"!= valid token total {valid_total}")
    started = run.get("generation_started_at")
    finished = run.get("generation_finished_at")
    if (not isinstance(started, (int, float))
            or isinstance(started, bool)
            or not isinstance(finished, (int, float))
            or isinstance(finished, bool) or finished < started):
        errors.append(f"{name}: invalid generation window")
    return errors


def validate_c2_result(result: dict) -> dict:
    """对 sigmoid C2 证据做严格、可单测的验收判定。"""
    errors: list[str] = []
    runs = result.get("runs")
    if not isinstance(runs, dict):
        runs = {}
        errors.append("runs missing")
    baseline = runs.get("baseline")
    injected = runs.get("injected")
    errors.extend(_validate_run("baseline", baseline))
    errors.extend(_validate_run("injected", injected))
    baseline = baseline if isinstance(baseline, dict) else {}
    injected = injected if isinstance(injected, dict) else {}

    baseline_run_id = baseline.get("run_id")
    injected_run_id = injected.get("run_id")
    if baseline_run_id == injected_run_id:
        errors.append("baseline/injected run_id must be distinct")
    if baseline.get("injection_enabled") is not False:
        errors.append("baseline inherited injection switch")
    if injected.get("injection_enabled") is not True:
        errors.append("injected run did not enable injection")

    base_tokens = baseline.get("output_token_ids")
    injected_tokens = injected.get("output_token_ids")
    output_match = bool(
        result.get("output_match") is True
        and isinstance(base_tokens, list)
        and base_tokens == injected_tokens
    )
    if not output_match:
        errors.append("baseline/injected generated tokens mismatch")

    for key in ("sigmoid_consumer_errors", "sigmoid_counter_errors",
                "baseline_counter_errors", "baseline_consumer_errors"):
        value = result.get(key, [])
        if not isinstance(value, list) or value:
            errors.append(f"{key}: {value!r}")
    if result.get("baseline_counter_records"):
        errors.append("baseline produced injected counter evidence")
    if result.get("baseline_consumer_records"):
        errors.append("baseline produced injected consumer evidence")

    started = injected.get("generation_started_at")
    finished = injected.get("generation_finished_at")
    valid_window = (
        isinstance(started, (int, float)) and not isinstance(started, bool)
        and isinstance(finished, (int, float))
        and not isinstance(finished, bool) and finished >= started
    )
    runner_pid = injected.get("runner_pid")
    generation_hits = 0
    hit_pids: set[int] = set()
    counters = result.get("sigmoid_counter_records")
    if not isinstance(counters, list) or not counters:
        errors.append("sigmoid counter evidence missing")
        counters = []
    for index, record in enumerate(counters):
        prefix = f"counter[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix}: expected object")
            continue
        if record.get("op") != "sigmoid":
            errors.append(f"{prefix}: op mismatch")
        if record.get("run_id") != injected_run_id:
            errors.append(f"{prefix}: run_id mismatch")
        pid = record.get("pid")
        if not _positive_int(pid):
            errors.append(f"{prefix}: pid invalid")
            continue
        if pid == runner_pid:
            errors.append(f"{prefix}: hit came from outer runner")
        hit_times = record.get("hit_times")
        hits = record.get("hits")
        if (not isinstance(hit_times, list)
                or any(not isinstance(t, (int, float))
                       or isinstance(t, bool) for t in hit_times)):
            errors.append(f"{prefix}: hit_times invalid")
            continue
        if (not isinstance(hits, int) or isinstance(hits, bool) or hits < 0
                or hits != len(hit_times)):
            errors.append(f"{prefix}: hits do not match hit_times")
        in_generation = (
            sum(started <= t <= finished for t in hit_times)
            if valid_window else 0
        )
        if in_generation:
            generation_hits += in_generation
            hit_pids.add(pid)
    if generation_hits <= 0:
        errors.append("no sigmoid hit increment in generation window")

    consumers = result.get("sigmoid_consumer")
    if not isinstance(consumers, list) or not consumers:
        errors.append("sigmoid consumer evidence missing")
        consumers = []
    matched_consumers = 0
    for index, record in enumerate(consumers):
        prefix = f"consumer[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix}: expected object")
            continue
        expected_fields = {
            "type": "consume",
            "consumer_type": "sampler_logits",
            "consumer": "vllm.v1.sample.Sampler.forward",
            "phase": "generation",
            "run_id": injected_run_id,
        }
        for key, expected in expected_fields.items():
            if record.get(key) != expected:
                errors.append(f"{prefix}: {key} mismatch")
        pid = record.get("pid")
        timestamp = record.get("timestamp")
        pid_matches = _positive_int(pid) and pid in hit_pids
        time_matches = (
            valid_window and isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and started <= timestamp <= finished
        )
        if not pid_matches:
            errors.append(f"{prefix}: pid has no generation hit")
        if not time_matches:
            errors.append(f"{prefix}: timestamp outside generation window")
        if (pid_matches and time_matches
                and all(record.get(k) == v
                        for k, v in expected_fields.items())):
            matched_consumers += 1
    if matched_consumers <= 0:
        errors.append("no consumer record matches run/PID/generation phase")

    sigmoid_hits = result.get("counts", {}).get("sigmoid", 0)
    if not _positive_int(sigmoid_hits):
        errors.append("aggregate sigmoid hit count is zero/invalid")
    return {
        "passed": not errors,
        "errors": errors,
        "output_match": output_match,
        "expected_requests": injected.get("expected_request_count", 0),
        "generated_requests": injected.get("generated_request_count", 0),
        "generated_tokens": injected.get("generated_token_count", 0),
        "sigmoid_hits": sigmoid_hits,
        "generation_hits": generation_hits,
        "worker_process_hit": bool(hit_pids),
        "consumer_records": len(consumers),
        "matched_consumer_records": matched_consumers,
    }


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
| 生成阶段命中增量 | `{acceptance['generation_hits']}` |
| 请求 / token | `{acceptance['generated_requests']}/{acceptance['expected_requests']}` 请求；`{acceptance['generated_tokens']}` token |
| worker 命中 | `{acceptance['worker_process_hit']}`（命中 PID 与 runner PID 分离） |
| 输出一致 | `{acceptance['output_match']}` |
| 证据错误 | `{json.dumps(acceptance['errors'], ensure_ascii=False)}` |

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

    acceptance = validate_c2_result(result)
    passed = acceptance["passed"]

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
            "visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", profile.visible_devices),
            "profile_visible_devices": profile.visible_devices,
            "tensor_parallel_size": profile.framework.get(
                "tensor_parallel_size"),
            "distributed_executor_backend": profile.framework.get(
                "distributed_executor_backend"),
        },
        "result": result,
        "acceptance": acceptance,
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
