# 统一性能基准与回归记录
#
# 设计要点:
#   - 计时口径与样例一致: 短采样(warmup + iters<=100) + synchronize
#   - 记录为结构化 JSON(含 git commit + 环境指纹), 供基线对比
#   - 运行记录落 results/perf/runs/<device>/(不入库), 基线在 perf/baselines/(入库)
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = ROOT / "perf" / "baselines"
RUN_DIR = ROOT / "results" / "perf" / "runs"
SCHEMA = 1


# ============ 用例与记录 ============

@dataclass
class PerfCase:
    """一个可重复执行的性能用例。

    make_fn(profile) 返回零参可调用（此时分配输入张量）；
    derived(latency_ms) 由延迟派生带宽/算力等附加指标。
    """

    case_id: str
    group: str                       # example | matrix-kernel | intake
    level: str                       # kernel / op / example ...
    make_fn: Callable                # profile -> () -> Any
    derived: Optional[Callable[[float], dict]] = None
    label: str = ""
    warmup: int = 20
    iters: int = 100


@dataclass
class PerfRecord:
    case_id: str
    group: str
    level: str
    device: str
    metrics: dict
    warmup: int
    iters: int
    timestamp: str
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    env_fingerprint: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "case_id": self.case_id,
            "group": self.group,
            "level": self.level,
            "device": self.device,
            "metrics": self.metrics,
            "warmup": self.warmup,
            "iters": self.iters,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "env_fingerprint": self.env_fingerprint,
        }


# ============ 计时 ============

def _sync(device: Optional[str]) -> None:
    """按设备类型同步；P800(XPU 经 XMLIR 伪装 cuda) 与现有样例口径一致。"""
    if device and device.startswith("cpu"):
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def bench(fn: Callable, *, warmup: int = 20, iters: int = 100,
          device: Optional[str] = None) -> float:
    """短采样计时，返回单次调用延迟(ms)。"""
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def run_case(case: PerfCase, profile) -> PerfRecord:
    """执行一个用例并返回结构化记录（不落盘，由调用方决定去向）。"""
    fn = case.make_fn(profile)
    latency = bench(fn, warmup=case.warmup, iters=case.iters,
                    device=profile.torch_device)
    metrics = {"latency_ms": round(latency, 6)}
    if case.derived:
        metrics.update({k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in case.derived(latency).items()})
    commit, dirty = git_state()
    return PerfRecord(
        case_id=case.case_id, group=case.group, level=case.level,
        device=profile.name, metrics=metrics,
        warmup=case.warmup, iters=case.iters,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=commit, git_dirty=dirty,
        env_fingerprint=env_fingerprint(),
    )


# ============ 元信息 ============

def git_state() -> tuple[Optional[str], Optional[bool]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return commit, bool(status)
    except Exception:
        return None, None


def env_fingerprint() -> dict:
    """影响性能可比性的最小软件栈指纹。"""
    out = {}
    for pkg in ("torch", "triton", "flag_gems", "vllm_fl", "vllm"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[pkg] = None
    return out


# ============ 落盘 ============

def _safe_name(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in case_id)


def classify_delta(delta: float | None, warn: float, fail: float) -> str:
    """按回归阈值分级: OK / WARN / FAIL（delta 为相对变慢比例）。"""
    if delta is None or delta <= warn:
        return "OK"
    return "WARN" if delta <= fail else "FAIL"


def save_run_record(rec: PerfRecord) -> Path:
    d = RUN_DIR / rec.device
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_safe_name(rec.case_id)}.json"
    p.write_text(json.dumps(rec.to_dict(), indent=2))
    return p


def load_run_records(device: str) -> list[dict]:
    d = RUN_DIR / device
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return out


def baseline_path(device: str) -> Path:
    return PERF_DIR / f"{device}.json"


def load_baseline(device: str) -> Optional[dict]:
    p = baseline_path(device)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def save_baseline(device: str, records: list[PerfRecord], *,
                  merge: bool = True) -> Path:
    """写入/合并基线。merge=True 时保留本次未运行用例的旧基线值。"""
    return save_baseline_dicts(
        device, [r.to_dict() if isinstance(r, PerfRecord) else r
                 for r in records], merge=merge)


def save_baseline_dicts(device: str, records: list[dict], *,
                        merge: bool = True) -> Path:
    p = baseline_path(device)
    old = json.loads(p.read_text()) if (merge and p.is_file()) else {}
    cases = old.get("cases", {})
    for r in records:
        cases[r["case_id"]] = {
            "metrics": r["metrics"], "warmup": r["warmup"],
            "iters": r["iters"], "updated_at": r["timestamp"],
            "git_commit": r["git_commit"], "git_dirty": r["git_dirty"],
            "env_fingerprint": r["env_fingerprint"],
        }
    commit, dirty = git_state()
    bundle = {
        "schema": SCHEMA,
        "device": device,
        "created_at": old.get("created_at", records[0]["timestamp"] if records else
                              datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "git_dirty": dirty,
        "env_fingerprint": env_fingerprint(),
        "cases": dict(sorted(cases.items())),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle, indent=2))
    return p
