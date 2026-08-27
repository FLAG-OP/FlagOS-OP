"""性能记录/基线/门禁分级的纯逻辑单测（无需 GPU）。"""
import json

import common.perf as P


def test_classify_delta():
    assert P.classify_delta(0.0, 0.2, 0.3) == "OK"
    assert P.classify_delta(0.2, 0.2, 0.3) == "OK"      # 恰在 WARN 线内
    assert P.classify_delta(0.21, 0.2, 0.3) == "WARN"
    assert P.classify_delta(0.30, 0.2, 0.3) == "WARN"   # 恰在 FAIL 线内
    assert P.classify_delta(0.31, 0.2, 0.3) == "FAIL"
    assert P.classify_delta(-0.5, 0.2, 0.3) == "OK"     # 变快
    assert P.classify_delta(None, 0.2, 0.3) == "OK"     # 无基线值


def _rec(cid, ms):
    return {"case_id": cid, "metrics": {"latency_ms": ms},
            "warmup": 20, "iters": 100, "timestamp": "2026-01-01T00:00:00",
            "git_commit": "x", "git_dirty": False, "env_fingerprint": {}}


def test_baseline_merge_preserves_unrun(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PERF_DIR", tmp_path)
    P.save_baseline_dicts("dev", [_rec("a", 1.0), _rec("b", 2.0)])
    # 只重跑 a: b 的旧基线应保留
    P.save_baseline_dicts("dev", [_rec("a", 1.5)])
    cases = json.loads((tmp_path / "dev.json").read_text())["cases"]
    assert cases["a"]["metrics"]["latency_ms"] == 1.5
    assert cases["b"]["metrics"]["latency_ms"] == 2.0


def test_run_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RUN_DIR", tmp_path)
    rec = P.PerfRecord(case_id="x.y-z", group="example", level="kernel",
                       device="dev", metrics={"latency_ms": 0.5},
                       warmup=20, iters=100, timestamp="2026-01-01T00:00:00")
    P.save_run_record(rec)
    loaded = P.load_run_records("dev")
    assert len(loaded) == 1 and loaded[0]["metrics"]["latency_ms"] == 0.5
