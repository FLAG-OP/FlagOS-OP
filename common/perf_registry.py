# 性能用例注册表: 汇总样例与矩阵 kernel 层的 perf_cases()
from __future__ import annotations

import importlib.util
from pathlib import Path

from common.perf import PerfCase

ROOT = Path(__file__).resolve().parents[1]

# (提供者 id, 文件路径)。模块需实现 perf_cases(profile) -> list[PerfCase]
PROVIDERS = [
    ("matrix.a1", ROOT / "tests/kernel_level/test_a1.py"),
    ("matrix.a2", ROOT / "tests/kernel_level/test_a2.py"),
    ("matrix.b", ROOT / "tests/kernel_level/test_b.py"),
    ("example.a2-op", ROOT / "examples/a2-op/example.py"),
    ("example.b-fullstack", ROOT / "examples/b-fullstack/example.py"),
    ("example.bmm-fullstack", ROOT / "examples/bmm-fullstack/example.py"),
    ("example.softmax-fullstack", ROOT / "examples/softmax-fullstack/example.py"),
    ("example.backward-example", ROOT / "examples/backward-example/example.py"),
    ("example.hw-kernel-example", ROOT / "examples/hw-kernel-example/example.py"),
    ("ops.sdpa", ROOT / "ops/sdpa/example.py"),
    ("ops.embedding", ROOT / "ops/embedding/example.py"),
]

_cache: dict[str, object] = {}


def _load(provider_id: str, path: Path):
    if provider_id in _cache:
        return _cache[provider_id]
    name = "perf_provider_" + provider_id.replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cache[provider_id] = mod
    return mod


def available_providers() -> list[str]:
    return [pid for pid, _ in PROVIDERS]


def load_cases(profile, *, groups: list[str] | None = None,
               pattern: str | None = None) -> list[PerfCase]:
    """加载全部（或按 group / case_id 子串过滤）性能用例。

    单个提供者缺失 perf_cases 或导入失败时跳过并提示，不影响其余用例。
    """
    cases: list[PerfCase] = []
    for pid, path in PROVIDERS:
        try:
            mod = _load(pid, path)
            fn = getattr(mod, "perf_cases", None)
            if fn is None:
                print(f"  [perf-registry] {pid}: 无 perf_cases，跳过")
                continue
            cases.extend(fn(profile))
        except Exception as e:
            print(f"  [perf-registry] {pid}: 加载失败跳过 ({e})")
    if groups:
        cases = [c for c in cases if c.group in groups]
    if pattern:
        cases = [c for c in cases if pattern in c.case_id]
    return cases
