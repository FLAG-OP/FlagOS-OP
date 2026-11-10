"""Unit tests for scripts/c2_enginecore/validate.py::decide (negative cases).

Covers the failure modes the reviewer called out: init-only (generation hits 0),
empty output, token mismatch, subprocess failure, missing phase dump.
"""
import importlib.util
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "c2_enginecore" / "validate.py"
_spec = importlib.util.spec_from_file_location("c2_enginecore_validate", _P)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
decide = _mod.decide
aggregate = _mod._agg


def _valid():
    return {
        "baseline_rc": 0, "injected_full_rc": 0, "injected_init_rc": 0,
        "baseline_ids": [1, 2, 3], "injected_ids": [1, 2, 3],
        "has_full_dump": True, "generation_hits": 97,
        "init_only": {"exec_calls": 0, "gen": 0},
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


def test_init_only_forward_fails():
    s = _valid()
    s["init_only"]["exec_calls"] = 1
    ok, reasons = decide(s)
    assert not ok and any("init-only run executed" in r for r in reasons)


def test_evidence_error_fails_closed():
    s = _valid()
    s["evidence_errors"] = ["foreign run_id"]
    ok, reasons = decide(s)
    assert not ok and any("invalid phase evidence" in r for r in reasons)


def test_required_performance_evidence_fails_when_not_collected():
    s = _valid()
    s["performance_required"] = True
    s["performance"] = {"baseline": {"status": "not-collected"},
                         "injected": {"status": "not-collected"}}
    ok, reasons = decide(s)
    assert not ok and any("performance evidence missing" in r for r in reasons)


def _record(run_id="run-1", **overrides):
    record = {
        "run_id": run_id, "role": "unknown", "pid": 1, "ppid": 0,
        "patched": True, "ops": ["empty_like"], "platform": "ascend",
        "dispatch_key": "AutogradPrivateUse1",
        "init_count": {"empty_like": 2},
        "generation_hits": {"empty_like": 3},
        "final_count": {"empty_like": 5}, "execute_calls": 1,
    }
    record.update(overrides)
    return record


def test_aggregate_rejects_foreign_or_malformed_evidence(tmp_path):
    path = tmp_path / "count_1.json"
    path.write_text(json.dumps(_record(run_id="other")))
    try:
        aggregate(tmp_path, "empty_like", "run-1")
    except ValueError as exc:
        assert "run_id" in str(exc)
    else:
        raise AssertionError("foreign evidence was accepted")


def test_aggregate_rejects_phase_sum_mismatch(tmp_path):
    record = _record()
    record["final_count"]["empty_like"] = 99
    (tmp_path / "count_1.json").write_text(json.dumps(record))
    try:
        aggregate(tmp_path, "empty_like", "run-1")
    except ValueError as exc:
        assert "final != init + generation" in str(exc)
    else:
        raise AssertionError("inconsistent evidence was accepted")
