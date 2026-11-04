"""Unit tests for scripts/c2_enginecore/validate.py::decide (negative cases).

Covers the failure modes the reviewer called out: init-only (generation hits 0),
empty output, token mismatch, subprocess failure, missing phase dump.
"""
import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "c2_enginecore" / "validate.py"
_spec = importlib.util.spec_from_file_location("c2_enginecore_validate", _P)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
decide = _mod.decide


def _valid():
    return {
        "baseline_rc": 0, "injected_full_rc": 0, "injected_init_rc": 0,
        "baseline_ids": [1, 2, 3], "injected_ids": [1, 2, 3],
        "has_full_dump": True, "generation_hits": 97,
    }


def test_valid_passes():
    ok, reasons = decide(_valid())
    assert ok and not reasons


def test_init_only_generation_hits_zero_fails():
    s = _valid()
    s["generation_hits"] = 0
    ok, reasons = decide(s)
    assert not ok and any("generation-phase hits" in r for r in reasons)


def test_empty_output_fails():
    s = _valid()
    s["baseline_ids"] = []
    ok, reasons = decide(s)
    assert not ok and any("empty token ids" in r for r in reasons)


def test_mismatch_fails():
    s = _valid()
    s["injected_ids"] = [1, 2, 4]
    ok, reasons = decide(s)
    assert not ok and any("mismatch" in r for r in reasons)


def test_subprocess_failure_fails():
    s = _valid()
    s["injected_full_rc"] = 1
    ok, reasons = decide(s)
    assert not ok and any("failed" in r for r in reasons)


def test_missing_phase_dump_fails():
    s = _valid()
    s["has_full_dump"] = False
    ok, reasons = decide(s)
    assert not ok and any("phase dump" in r for r in reasons)
