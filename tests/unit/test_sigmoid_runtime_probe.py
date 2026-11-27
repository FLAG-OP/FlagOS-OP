"""P800 分阶段环境 probe 的纯 CPU 契约测试。"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "ops/sigmoid/script/p800_runtime_probe.py"
SPEC = importlib.util.spec_from_file_location("sigmoid_runtime_probe", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PROBE)


def test_stage_names_are_unique_and_cover_required_boundaries():
    stages = PROBE.stage_specs()
    names = [stage["name"] for stage in stages]
    assert len(names) == len(set(names))
    assert {
        "identity", "copy_17", "copy_255", "copy_14336",
        "native_sigmoid_17", "native_sigmoid_255",
        "native_sigmoid_14336", "generic_triton_257",
        "sigmoid_257", "sigmoid_255", "vllm_tp2_c2",
    } == set(names)
    assert all(stage["timeout_s"] > 0 for stage in stages)


def test_summary_reports_first_failure_and_timeout_separately():
    summary = PROBE.summarize([
        {"name": "identity", "status": "PASS"},
        {"name": "copy_17", "status": "FAIL"},
        {"name": "copy_255", "status": "TIMEOUT"},
    ])
    assert summary == {
        "diagnostic_status": "FAIL",
        "passed": 1,
        "failed": 1,
        "timed_out": 1,
        "first_failure": "copy_17",
        "note": "diagnostic-only; PASS does not constitute C1/C2 acceptance",
    }


def test_summary_pass_is_still_diagnostic_only():
    summary = PROBE.summarize([
        {"name": "identity", "status": "PASS"},
    ])
    assert summary["diagnostic_status"] == "PASS"
    assert "does not constitute" in summary["note"]
