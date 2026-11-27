"""sigmoid 真实 vLLM C2 的生成阶段证据与环境隔离回归测试。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


C2 = _load("sigmoid_c2_contract", "ops/sigmoid/script/c2_real_vllm.py")
HARNESS = _load("sigmoid_c2_harness", "tests/framework_level/_aten_harness.py")


def _run(run_id: str, *, injected: bool, pid: int) -> dict:
    return {
        "runner_pid": pid,
        "run_id": run_id,
        "injection_enabled": injected,
        "expected_request_count": 2,
        "generated_request_count": 2,
        "generated_token_count": 3,
        "generation_started_at": 10.0,
        "generation_finished_at": 20.0,
        "output_token_ids": [[1, 2], [3]],
    }


def _good_result() -> dict:
    return {
        "output_match": True,
        "counts": {"sigmoid": 2},
        "runs": {
            "baseline": _run("baseline-run", injected=False, pid=100),
            "injected": _run("injected-run", injected=True, pid=200),
        },
        "sigmoid_counter_records": [{
            "op": "sigmoid",
            "run_id": "injected-run",
            "pid": 300,
            "hits": 2,
            "hit_times": [12.0, 13.0],
        }],
        "sigmoid_counter_errors": [],
        "sigmoid_consumer": [{
            "type": "consume",
            "consumer_type": "sampler_logits",
            "consumer": "vllm.v1.sample.Sampler.forward",
            "phase": "generation",
            "run_id": "injected-run",
            "pid": 300,
            "timestamp": 12.5,
        }],
        "sigmoid_consumer_errors": [],
        "baseline_counter_records": [],
        "baseline_counter_errors": [],
        "baseline_consumer_records": [],
        "baseline_consumer_errors": [],
    }


def test_generation_evidence_with_matching_run_pid_and_consumer_passes():
    acceptance = C2.validate_c2_result(_good_result())
    assert acceptance["passed"] is True
    assert acceptance["generation_hits"] == 2
    assert acceptance["matched_consumer_records"] == 1


def test_empty_outputs_cannot_pass_with_positive_process_hits():
    result = _good_result()
    for run in result["runs"].values():
        run["generated_request_count"] = 0
        run["generated_token_count"] = 0
        run["output_token_ids"] = []
    acceptance = C2.validate_c2_result(result)
    assert acceptance["passed"] is False
    assert any("generated 0/2 requests" in error
               for error in acceptance["errors"])


def test_initialization_hits_do_not_count_as_generation_hits():
    result = _good_result()
    result["sigmoid_counter_records"][0].update({
        "hits": 2,
        "hit_times": [8.0, 9.0],
    })
    acceptance = C2.validate_c2_result(result)
    assert acceptance["passed"] is False
    assert acceptance["generation_hits"] == 0
    assert any("no sigmoid hit increment" in error
               for error in acceptance["errors"])


def test_unrelated_consumer_run_and_pid_cannot_satisfy_acceptance():
    result = _good_result()
    result["sigmoid_consumer"][0].update({
        "run_id": "someone-else",
        "pid": 999,
    })
    acceptance = C2.validate_c2_result(result)
    assert acceptance["passed"] is False
    assert acceptance["matched_consumer_records"] == 0
    assert any("run_id mismatch" in error for error in acceptance["errors"])
    assert any("pid has no generation hit" in error
               for error in acceptance["errors"])


def test_corrupt_or_missing_key_evidence_fails_closed():
    result = _good_result()
    result["sigmoid_consumer_errors"] = ["line 2: broken JSON"]
    del result["sigmoid_counter_records"][0]["run_id"]
    acceptance = C2.validate_c2_result(result)
    assert acceptance["passed"] is False
    assert any("sigmoid_consumer_errors" in error
               for error in acceptance["errors"])
    assert any("counter[0]: run_id mismatch" in error
               for error in acceptance["errors"])


def test_corrupt_jsonl_is_reported_instead_of_silently_dropped(tmp_path):
    evidence = tmp_path / "consumer.jsonl"
    evidence.write_text(
        '{"type": "consume"}\n{broken\n', encoding="utf-8")
    records, errors = HARNESS._jsonl_evidence(str(evidence))
    assert records == [{"type": "consume"}]
    assert len(errors) == 1
    assert "line 2" in errors[0]


def test_baseline_injection_or_counter_evidence_is_rejected():
    result = _good_result()
    result["runs"]["baseline"]["injection_enabled"] = True
    result["baseline_counter_records"] = [{"hits": 1}]
    acceptance = C2.validate_c2_result(result)
    assert acceptance["passed"] is False
    assert any("baseline inherited injection" in error
               for error in acceptance["errors"])
    assert any("baseline produced injected counter" in error
               for error in acceptance["errors"])


def test_run_clears_parent_injection_and_isolates_count_directories(
        tmp_path, monkeypatch):
    captured = []

    def fake_run(cmd, *, check, env, cwd):
        captured.append(env)
        Path(cmd[2]).write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr(HARNESS.subprocess, "run", fake_run)
    monkeypatch.setenv("FLAGOS_A1_ATEN_INJECT", "1")
    monkeypatch.setenv("A1_SIGMOID_COUNT_FILE", "/stale/count")

    base_dir = tmp_path / "baseline"
    inject_dir = tmp_path / "injected"
    base_dir.mkdir()
    inject_dir.mkdir()
    HARNESS._run("model", str(tmp_path / "base.json"), inject=False,
                 count_dir=str(base_dir), dispatch_key="CUDA",
                 run_id="base-run")
    HARNESS._run("model", str(tmp_path / "inject.json"), inject=True,
                 count_dir=str(inject_dir), dispatch_key="CUDA",
                 run_id="inject-run")

    baseline, injected = captured
    assert baseline["FLAGOS_A1_ATEN_INJECT"] == "0"
    assert baseline["FLAGOS_C2_COUNT_DIR"] == str(base_dir)
    assert baseline["FLAGOS_C2_RUN_ID"] == "base-run"
    assert "A1_SIGMOID_COUNT_FILE" not in baseline
    assert injected["FLAGOS_A1_ATEN_INJECT"] == "1"
    assert injected["FLAGOS_C2_COUNT_DIR"] == str(inject_dir)
    assert injected["FLAGOS_C2_RUN_ID"] == "inject-run"
    assert injected["A1_SIGMOID_COUNT_FILE"] == str(inject_dir / "sigmoid")
