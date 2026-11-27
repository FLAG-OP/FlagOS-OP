#!/usr/bin/env python3
"""分阶段采集 P800/XMLIR/sigmoid/vLLM 环境故障证据。

每个阶段使用独立子进程并带硬超时；即使前一阶段失败，后续阶段仍会
执行，以区分设备拷贝、原生算子、通用 Triton、sigmoid 专用路径和
真实 vLLM TP=2 初始化。输出仅用于环境诊断，不构成 C1/C2 验收结论。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


OP_DIR = Path(__file__).resolve().parents[1]
ROOT = OP_DIR.parent.parent
PROFILE = "p800-kunlunxin-gems5"
DEVICE = "cuda:1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_output(command: list[str], timeout_s: int = 15) -> dict:
    try:
        proc = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True,
            timeout=timeout_s, check=False,
        )
        return {
            "command": shlex.join(command),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": shlex.join(command),
            "returncode": 124 if isinstance(
                exc, subprocess.TimeoutExpired) else 127,
            "stdout": getattr(exc, "stdout", "") or "",
            "stderr": repr(exc),
        }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict:
    profile = ROOT / "configs/devices" / f"{PROFILE}.yaml"
    lock = ROOT / "configs/env" / f"{PROFILE}.lock.yaml"
    head = _command_output(["git", "rev-parse", "HEAD"])
    status = _command_output(["git", "status", "--short"])
    return {
        "collected_at": _utc_now(),
        "git_commit": head["stdout"].strip(),
        "git_status": status["stdout"].splitlines(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {
            "torch": _version("torch"),
            "triton": _version("triton"),
            "flag-gems": _version("flag-gems"),
            "vllm": _version("vllm"),
            "vllm-plugin-fl": _version("vllm-plugin-fl"),
            "torch-xmlir": _version("torch-xmlir"),
        },
        "profile": str(profile.relative_to(ROOT)),
        "profile_sha256": _sha256(profile),
        "env_lock": str(lock.relative_to(ROOT)),
        "env_lock_sha256": _sha256(lock),
        "visible_devices": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XPU_VISIBLE_DEVICES": os.environ.get("XPU_VISIBLE_DEVICES"),
        },
        "container_image": os.environ.get("FLAGOS_PROBE_CONTAINER_IMAGE"),
        "container_id": os.environ.get("HOSTNAME"),
        "driver_snapshot": _command_output(["xpu_smi"]),
        "device_nodes": _command_output(
            ["bash", "-lc", "ls -l /dev/xpu* 2>&1"]),
        "process_snapshot": _command_output(
            ["ps", "-eo", "pid,ppid,stat,lstart,cmd"]),
    }


def _worker_identity() -> None:
    import torch
    import torch_xmlir  # noqa: F401

    devices = []
    for index in range(torch.cuda.device_count()):
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
        })
    print(json.dumps({
        "pid": os.getpid(),
        "torch": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "selected_device": DEVICE,
    }, ensure_ascii=False), flush=True)


def _torch_sync(torch, device: str) -> None:
    torch.cuda.synchronize(torch.device(device))


def _worker_copy(numel: int) -> None:
    import torch
    import torch_xmlir  # noqa: F401

    source = torch.arange(numel, dtype=torch.float32)
    device_value = source.to(DEVICE)
    _torch_sync(torch, DEVICE)
    restored = device_value.cpu()
    _torch_sync(torch, DEVICE)
    print(json.dumps({
        "pid": os.getpid(), "stage": "device_copy", "numel": numel,
        "equal": bool(torch.equal(source, restored)),
        "last": float(restored[-1]) if numel else None,
    }), flush=True)


def _worker_native_sigmoid(numel: int) -> None:
    import torch
    import torch_xmlir  # noqa: F401

    value = torch.linspace(-8, 8, numel, dtype=torch.float32).to(DEVICE)
    result = torch.sigmoid(value)
    _torch_sync(torch, DEVICE)
    sample = result[[0, numel // 2, numel - 1]].cpu().tolist()
    print(json.dumps({
        "pid": os.getpid(), "stage": "native_sigmoid", "numel": numel,
        "sample": sample,
    }), flush=True)


def _worker_generic_triton(numel: int) -> None:
    import torch
    import torch_xmlir  # noqa: F401
    import triton
    import triton.language as tl

    @triton.jit
    def add_one_kernel(x_ptr, y_ptr, n_elements,
                       block_size: tl.constexpr):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        value = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, value + 1.0, mask=mask)

    value = torch.arange(numel, dtype=torch.float32, device=DEVICE)
    result = torch.empty_like(value)
    block = 256
    add_one_kernel[(triton.cdiv(numel, block),)](
        value, result, numel, block_size=block)
    _torch_sync(torch, DEVICE)
    sample = result[[0, numel - 1]].cpu().tolist()
    print(json.dumps({
        "pid": os.getpid(), "stage": "generic_triton", "numel": numel,
        "sample": sample,
    }), flush=True)


def _worker_sigmoid(numel: int) -> None:
    import torch
    import torch_xmlir  # noqa: F401

    from ops.sigmoid.kernel.triton_level import sigmoid_triton

    value = torch.linspace(-8, 8, numel, dtype=torch.float32).to(DEVICE)
    result = sigmoid_triton(value)
    _torch_sync(torch, DEVICE)
    sample = result[[0, numel // 2, numel - 1]].cpu().tolist()
    print(json.dumps({
        "pid": os.getpid(), "stage": "sigmoid_triton", "numel": numel,
        "sample": sample,
    }), flush=True)


def _run_worker(name: str, numel: int | None) -> None:
    if name == "identity":
        _worker_identity()
    elif name == "copy":
        assert numel is not None
        _worker_copy(numel)
    elif name == "native_sigmoid":
        assert numel is not None
        _worker_native_sigmoid(numel)
    elif name == "generic_triton":
        assert numel is not None
        _worker_generic_triton(numel)
    elif name == "sigmoid":
        assert numel is not None
        _worker_sigmoid(numel)
    else:
        raise ValueError(f"unknown worker stage: {name}")


def stage_specs() -> list[dict]:
    script = str(Path(__file__).resolve())
    python = sys.executable

    def worker(name: str, numel: int | None, timeout_s: int) -> dict:
        command = [python, script, "--worker", name]
        if numel is not None:
            command.extend(["--numel", str(numel)])
        suffix = "" if numel is None else f"_{numel}"
        return {
            "name": f"{name}{suffix}",
            "command": command,
            "timeout_s": timeout_s,
        }

    return [
        worker("identity", None, 30),
        worker("copy", 17, 30),
        worker("copy", 255, 30),
        worker("copy", 14336, 30),
        worker("native_sigmoid", 17, 30),
        worker("native_sigmoid", 255, 30),
        worker("native_sigmoid", 14336, 30),
        worker("generic_triton", 257, 60),
        worker("sigmoid", 257, 60),
        worker("sigmoid", 255, 60),
        {
            "name": "vllm_tp2_c2",
            "command": [
                python, str(OP_DIR / "script/c2_real_vllm.py"),
                "--profile", PROFILE,
                "--out", "/tmp/sigmoid-probe-c2.json",
                "--report", "/tmp/sigmoid-probe-c2.md",
            ],
            "timeout_s": 180,
        },
    ]


def _run_stage(spec: dict, out_dir: Path, env: dict[str, str]) -> dict:
    name = spec["name"]
    command = spec["command"]
    timeout_s = int(spec["timeout_s"])
    started_wall = time.time()
    started_at = _utc_now()
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
    ended_at = _utc_now()
    duration_s = round(time.time() - started_wall, 3)
    returncode = 124 if timed_out else proc.returncode
    stdout_path = out_dir / f"{name}.stdout.log"
    stderr_path = out_dir / f"{name}.stderr.log"
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")
    return {
        "name": name,
        "status": "TIMEOUT" if timed_out else (
            "PASS" if returncode == 0 else "FAIL"),
        "command": shlex.join(command),
        "pid": proc.pid,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "timeout_s": timeout_s,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout_path.name,
        "stdout_sha256": _sha256(stdout_path),
        "stderr": stderr_path.name,
        "stderr_sha256": _sha256(stderr_path),
    }


def summarize(stages: list[dict]) -> dict:
    failures = [stage for stage in stages if stage.get("status") != "PASS"]
    return {
        "diagnostic_status": "PASS" if not failures else "FAIL",
        "passed": sum(stage.get("status") == "PASS" for stage in stages),
        "failed": sum(stage.get("status") == "FAIL" for stage in stages),
        "timed_out": sum(
            stage.get("status") == "TIMEOUT" for stage in stages),
        "first_failure": failures[0].get("name") if failures else None,
        "note": (
            "diagnostic-only; PASS does not constitute C1/C2 acceptance"
        ),
    }


def _write_report(out_dir: Path, manifest: dict) -> None:
    rows = []
    for stage in manifest["stages"]:
        rows.append(
            f"| `{stage['name']}` | {stage['status']} | "
            f"{stage['returncode']} | {stage['duration_s']} | "
            f"[{stage['stdout']}]({stage['stdout']}) / "
            f"[{stage['stderr']}]({stage['stderr']}) |"
        )
    summary = manifest["summary"]
    environment = manifest["environment"]
    text = f"""# aten.sigmoid P800 运行时分阶段故障材料

> 该材料仅用于环境诊断，不构成 C1/C2 或性能验收结论。

| 项 | 值 |
|---|---|
| 采集时间 | `{environment['collected_at']}` |
| Git SHA | `{environment['git_commit']}` |
| profile | `{manifest['profile']}` |
| 可见设备 | `{json.dumps(environment['visible_devices'], ensure_ascii=False)}` |
| 容器镜像 | `{environment['container_image']}` |
| 诊断结果 | **{summary['diagnostic_status']}** |
| 首个失败阶段 | `{summary['first_failure']}` |
| 通过 / 失败 / 超时 | {summary['passed']} / {summary['failed']} / {summary['timed_out']} |

## 分阶段结果

| 阶段 | 状态 | 退出码 | 秒 | stdout / stderr |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

## 身份与完整性

- profile: `{environment['profile']}`，SHA256 `{environment['profile_sha256']}`
- env lock: `{environment['env_lock']}`，SHA256 `{environment['env_lock_sha256']}`
- manifest: [manifest.json](manifest.json)
- 设备快照、进程快照和每个日志的 SHA256 均在 manifest 中。
"""
    (out_dir / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=(
        "identity", "copy", "native_sigmoid", "generic_triton", "sigmoid"))
    parser.add_argument("--numel", type=int)
    parser.add_argument(
        "--out-dir",
        default=str(OP_DIR / "evidence/p800-runtime-failure-2026-11-27"),
    )
    args = parser.parse_args()
    if args.worker:
        _run_worker(args.worker, args.numel)
        return 0

    environment = _environment()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "1,2",
        "XPU_VISIBLE_DEVICES": "1,2",
    })
    stages = []
    for spec in stage_specs():
        print(f"[probe] start {spec['name']}", flush=True)
        stage = _run_stage(spec, out_dir, env)
        stages.append(stage)
        print(
            f"[probe] {stage['name']} -> {stage['status']} "
            f"rc={stage['returncode']} {stage['duration_s']}s",
            flush=True,
        )
    after = {
        "collected_at": _utc_now(),
        "driver_snapshot": _command_output(["xpu_smi"]),
        "process_snapshot": _command_output(
            ["ps", "-eo", "pid,ppid,stat,lstart,cmd"]),
    }
    manifest = {
        "schema": 1,
        "operator": "aten::sigmoid",
        "artifact_type": "environment-failure-diagnostic",
        "acceptance_role": "background",
        "profile": PROFILE,
        "device": DEVICE,
        "environment": environment,
        "stages": stages,
        "summary": summarize(stages),
        "after": after,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, manifest)
    print(f"[probe] evidence: {out_dir}")
    return 0 if manifest["summary"]["diagnostic_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
